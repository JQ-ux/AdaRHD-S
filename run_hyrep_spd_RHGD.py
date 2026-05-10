# structure of the code is derived from https://github.com/andyjm3/rhgd
import torch
import geoopt
import numpy as np

import os
import sys
from datetime import datetime

notebook_dir = os.getcwd()
rhgd_dir = notebook_dir + '/rhgd'
sys.path.append(rhgd_dir)

from geoopt import linalg, ManifoldParameter
from geoopt import SymmetricPositiveDefinite
from geoopt import Stiefel

import argparse
from rhgd.manifolds import EuclideanMod
from rhgd.utils import autograd, compute_hypergrad2
from rhgd.optimizer import RHGDstep
import pickle

## Euclidean ndim important
## ns approximation sensitive to conditioning of the Hessian and need to take very long step

def vec(X):
    """Reshape a symmetric matrix into a vector by extracting its upper-triangular part"""
    d = X.shape[-1]
    return X[..., torch.triu_indices(d, d)[0], torch.triu_indices(d, d)[1]]
 
def shallowSPDnet(hparams, params, A):
    W = hparams[0]
    gamma = params[0]
    Z = W.transpose(-1, -2) @ A @ W
    Z = linalg.sym_logm(Z)
    Z = vec(Z)
    return Z

def loss_lower(hparams, params, data):
    data_X, data_y = data
    gamma = params[0]
    pred = shallowSPDnet(hparams, params, data_X)
    loss = 0.5 * torch.norm(pred @ gamma - data_y) ** 2 / data_X.shape[0] + 0.5 * lam * torch.norm(gamma) ** 2
    return loss


def loss_upper(hparams, params, data):
    data_X, data_y = data
    gamma = params[0]
    pred = shallowSPDnet(hparams, params, data_X)
    loss = 0.5 * torch.norm(pred @ gamma - data_y) ** 2 / data_X.shape[0]
    return loss

def true_hessinv(loss_lower, hparams, params, data_lower, tangents):
    data_X, data_y = data_lower
    predg = shallowSPDnet(hparams, params, data_X)
    ehessg = (predg.transpose(-1, -2) @ predg) / data_X.shape[0] + lam * torch.eye(predg.shape[1], device=device)
    hessinvgrad = [torch.linalg.solve(ehessg, tangents[0])]
    return hessinvgrad

# Create a class to simultaneously write to stdout and a file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure output is written immediately
        
    def flush(self):
        # Needed for Python 3 compatibility
        self.terminal.flush()
        self.log.flush()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eta_x', type=float, default=0.05)
    parser.add_argument('--eta_y', type=float, default=0.05)
    parser.add_argument('--lower_iter', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--hygrad_opt', type=str, default='cg', choices=['hinv', 'cg', 'ns', 'ad'])
    parser.add_argument('--ns_gamma', type=float, default=0.05)
    parser.add_argument('--ns_iter', type=int, default=50)
    parser.add_argument('--cg_gamma', type=float, default=0.)
    parser.add_argument('--cg_iter', type=int, default=50)
    parser.add_argument('--compute_hg_error', type=bool, default=True)
    parser.add_argument('--stochastic', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--N', type=int, default=200)
    parser.add_argument('--d', type=int, default=50)
    parser.add_argument('--r', type=int, default=10)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--log', type=bool, default=False)
    args = parser.parse_args()

    if args.log:
        # Create a log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        log_filename = f"hyrep_spd_RHGD_log_{timestamp}.txt"
        sys.stdout = Logger(log_filename)

    # Generate random seeds for multiple runs
    np.random.seed(args.seed)
    seeds = np.random.randint(0, 10000, size=args.runs).tolist()
    print(f"Using seeds: {seeds}")
    print(f"N: {args.N}, d: {args.d}, r: {args.r}")

    epochs_all_runs = []
    runtime_all_runs = []
    loss_u_all_runs = []
    hg_error_all_runs = []
    hg_norm_all_runs = []

    for run in range(args.runs):

        # set up
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(seeds[run])
        print(device)
        torch.random.manual_seed(seeds[run])
        np.random.seed(seeds[run])
        torch.backends.cudnn.deterministic = True

        # init
        mfd = SymmetricPositiveDefinite()
        N = args.N
        d = args.d
        r = args.r
        lam = 0.1

        # generate A matrices
        A = mfd.random(N, d, d, device=device)  # N dxd SPD matrices
        Wstar = Stiefel().random(d, r, device=device)
        gammastar = torch.randn(int(r * (r + 1) / 2), device=device)

        y = Wstar.transpose(-1, -2) @ A @ Wstar
        y = linalg.sym_logm(y)
        y = vec(y) @ gammastar + torch.randn(N, device=device)

        Atr = A[:int(N/2)]
        Aval = A[int(N/2):]
        ytr = y[:int(N/2)]
        yval = y[int(N/2):]

        euclidean = EuclideanMod(ndim=1)
        stiefel = Stiefel(canonical=False)

        params = [geoopt.ManifoldParameter(euclidean.random(int(r * (r + 1) / 2), device=device), manifold=euclidean)]
        hparams = [geoopt.ManifoldParameter(torch.eye(d, r, device=device), manifold=stiefel)]
        mfd_params = [param.manifold for param in params]

        data_lower = [Atr, ytr]
        data_upper = [Aval, yval]

        # initial run
        for ii in range(args.lower_iter):
            if args.hygrad_opt == 'ad':
                grad = autograd(loss_lower(hparams, params, data_lower), params, create_graph=True)
                rgrad = [mfd.egrad2rgrad(param, egrad) for mfd, egrad, param in zip(mfd_params, grad, params)]
                params = [mfd.retr(param, - args.eta_y * rg) for mfd, param, rg in zip(mfd_params, params, rgrad)]
            else:
                grad = autograd(loss_lower(hparams, params, data_lower), params)
                with torch.no_grad():
                    for param, egrad in zip(params, grad):
                        rgrad = param.manifold.egrad2rgrad(param, egrad)
                        new_param = param.manifold.retr(param, -args.eta_y * rgrad)
                        param.copy_(new_param)

        params = [ManifoldParameter(p.detach().clone(), manifold=mfd) for mfd, p in zip(mfd_params, params)]

        true_hg = compute_hypergrad2(loss_lower, loss_upper, hparams, params,
                                    data_lower=data_lower, data_upper=data_upper,
                                    option='hinv', true_hessinv=true_hessinv,
                                    cg_iter=args.cg_iter, cg_gamma=args.cg_gamma,
                                    ns_gamma=args.ns_gamma, ns_iter=args.ns_iter)

        hypergrad = compute_hypergrad2(loss_lower, loss_upper, hparams, params,
                                    data_lower=data_lower, data_upper=data_upper,
                                    option=args.hygrad_opt, true_hessinv=true_hessinv,
                                    cg_iter=args.cg_iter, cg_gamma=args.cg_gamma,
                                    ns_gamma=args.ns_gamma, ns_iter=args.ns_iter)

        with torch.no_grad():
            hgradnorm = 0
            for hparam, hg in zip(hparams, hypergrad):
                hgradnorm += hparam.manifold.inner(hparam, hg).item() / len(hparams)
        hg_error = 0
        if not (args.hygrad_opt == 'hinv'):
            hg_error = [torch.sqrt(hp.manifold.inner(hp, hg - t_hg)).item() for hg, t_hg, hp in
                        zip(hypergrad, true_hg, hparams)]
            hg_error = torch.Tensor(hg_error).sum().item()
        epochs_all = [0]
        loss_u_all = [loss_upper(hparams, params, data_upper).item()]
        hg_norm_all = [hgradnorm]
        runtime = [0]
        hg_error_all = [hg_error]
        print(f"Epoch {0}: "
            f"loss upper: {loss_u_all[-1]:.4f}, "
            f"hypergrad norm: {hgradnorm:.4e},"
            f"hg error: {hg_error_all[-1]:.4e}")

        for ep in range(args.epoch):
            hparams, params, loss_u, hgradnorm, step_time, hg_error = RHGDstep(loss_lower, loss_upper, hparams, params, args,
                                                                               data=[data_lower, data_upper],
                                                                               true_hessinv=true_hessinv,
                                                                               )

            loss_u_all.append(loss_u)
            runtime.append(step_time)
            hg_error_all.append(hg_error)
            hg_norm_all.append(hgradnorm)
            epochs_all.append(ep)
            print(f"Epoch {ep}: "
                f"loss upper: {loss_u:.4e}, "
                f"hypergrad norm: {hgradnorm:.4e}, "
                f"hg error: {hg_error:.4e}, "
                f"step time: {step_time:.4f}")

            
        epochs_all_runs.append(torch.tensor(epochs_all).unsqueeze(0))
        runtime_all_runs.append(torch.tensor(runtime).unsqueeze(0))
        loss_u_all_runs.append(torch.tensor(loss_u_all).unsqueeze(0))
        hg_error_all_runs.append(torch.tensor(hg_error_all).unsqueeze(0))
        hg_norm_all_runs.append(torch.tensor(hg_norm_all).unsqueeze(0))

    stats = {'epochs': torch.cat(epochs_all_runs, dim=0), 'runtime': torch.cat(runtime_all_runs, dim=0), 
            "loss_upper": torch.cat(loss_u_all_runs, dim=0), "hg_error": torch.cat(hg_error_all_runs, dim=0), 
            'hg_norm': torch.cat(hg_norm_all_runs, dim=0)}
    
    res_folder = './results/'
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    filename = res_folder + 'shallow_hyrep_n' + str(N) + 'd' + str(d) + 'r' + str(r) + '_RHGD_lr' + str(args.eta_x) + '_loweriter' + str(args.lower_iter) + '.pickle'

    with open(filename, 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)