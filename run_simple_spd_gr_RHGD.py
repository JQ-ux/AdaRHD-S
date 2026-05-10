######## the experiment for Synthetic problem ########

####  max_{W in St(d,r)} tr(M* X^T Y W^T),  
#### s.t. M* = argmin_{M in S_{++}^d} <M, X^T X> + <M^{-1}, W Y^T Y W^T + ν I>,

# structure of the code is derived from https://github.com/andyjm3/rhgd


import argparse
import torch

import os
import sys
from datetime import datetime

notebook_dir = os.getcwd()
rhgd_dir = notebook_dir + '/rhgd'
sys.path.append(rhgd_dir)

import argparse
import torch

from geoopt import ManifoldParameter, Stiefel
from src.AdaRHD_manifolds import SymmetricPositiveDefiniteMod
from geoopt.linalg import sym_invm, sym_inv_sqrtm2
from scipy.linalg import solve_continuous_lyapunov
import numpy as np
from rhgd.utils import autograd, compute_hypergrad2
from rhgd.optimizer import RHGDstep
import pickle
import random

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def true_hessinv(loss_lower, hparams, params, data_lower, tangents):
    W = hparams[0]
    M = params[0]
    Minvhalf, Mhalf = sym_inv_sqrtm2(M)
    G = (tangents[0].T + tangents[0]) # there is a 2 multiplied
    # assert torch.allclose()
    A = (X.T @ X)
    B = (W @ (Y.T @ Y) @ W.T) + lam* torch.eye(d, device=W.device)
    lhs = Mhalf @ A @ Mhalf + Minvhalf @ B @ Minvhalf
    lhs = 0.5*(lhs + lhs.T)
    rhs = Minvhalf @ G @ Minvhalf
    U = solve_continuous_lyapunov(lhs.cpu().detach().numpy(), rhs.cpu().detach().numpy())
    U = torch.from_numpy(U).float().to(A.device)
    U = (U + U.transpose(-1,-2))/2
    U = Mhalf @ U @ Mhalf
    assert torch.allclose(U @ A @ M + M @ A @ U + U @ sym_invm(M) @ B + B @ sym_invm(M) @ U, G, atol=1e-5)
    return [U]


def loss_lower(hparams, params, data=None):
    W = hparams[0]
    M = params[0]
    Minv = sym_invm(M)
    loss = (M * (X.T @ X)).sum() + (Minv * (W @ (Y.T @ Y) @ W.T)).sum() + lam * torch.trace(Minv)
    return loss


def loss_upper(hparams, params, data=None):
    W = hparams[0]
    M = params[0]
    # Mhalf = sym_sqrtm(M)
    # loss = -0.5 * torch.trace(W.T @ (Mhalf @ (X.T @ X) @ Mhalf) @ W)
    loss = -torch.trace(M @ X.T @ Y @ W.T)
    # loss = torch.norm(X @ M - Y @ W.T, p='fro')**2
    return loss

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
    parser.add_argument('--eta_x', type=float, default=0.5)
    parser.add_argument('--eta_y', type=float, default=0.5)
    parser.add_argument('--lower_iter', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--hygrad_opt', type=str, default='cg', choices=['hinv', 'cg', 'ns', 'ad'])
    parser.add_argument('--cg_gamma', type=float, default=0.)
    parser.add_argument('--cg_iter', type=int, default=50)
    parser.add_argument('--ns_gamma', type=float, default=0.1)
    parser.add_argument('--ns_iter', type=int, default=50)
    parser.add_argument('--compute_hg_error', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--d', type=int, default=50)
    parser.add_argument('--r', type=int, default=20)
    parser.add_argument('--log', type=bool, default=False)
    args = parser.parse_args()

    if args.log:
        # Create a log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        log_filename = f"simple_spd_gr_RHGD_log_{timestamp}.txt"
        sys.stdout = Logger(log_filename)

    # Generate random seeds for multiple runs
    np.random.seed(args.seed)
    seeds = np.random.randint(0, 10000, size=args.runs).tolist()
    print(f"Using seeds: {seeds}")
    print(f"n: {args.n}, d: {args.d}, r: {args.r}")

    epochs_all_runs = []
    runtime_all_runs = []
    loss_u_all_runs = []
    hg_error_all_runs = []
    hg_norm_all_runs = []

    for run in range(args.runs):

        # set up
        seed = seeds[run]
        print(f"Run {run+1}/{args.runs} with seed {seed}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(seed)
        print(device)
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True

        n = args.n
        d = args.d
        r = args.r
        lam = 0.01

        X = torch.randn(n, d, device=device)
        X = X/torch.norm(X)
        Y = torch.randn(n, r, device=device)
        Y = Y/torch.norm(Y)

        spd = SymmetricPositiveDefiniteMod()
        st = Stiefel(canonical=False)
        hparams = [ManifoldParameter(st.random(d,r, device=device), manifold=st)]
        params = [ManifoldParameter(torch.eye(d, device=device), manifold=spd)]
        mfd_params = [spd]

        # initial run
        for ii in range(args.lower_iter):
            if args.hygrad_opt == 'ad':
                grad = autograd(loss_lower(hparams, params, None), params, create_graph=True)
                rgrad = [mfd.egrad2rgrad(param, egrad) for mfd, egrad, param in zip(mfd_params, grad, params)]
                params = [mfd.retr(param, - args.eta_y * rg) for mfd, param, rg in zip(mfd_params, params, rgrad)]
            else:
                grad = autograd(loss_lower(hparams, params, None), params)
                with torch.no_grad():
                    for param, egrad in zip(params, grad):
                        rgrad = param.manifold.egrad2rgrad(param, egrad)
                        new_param = param.manifold.retr(param, -args.eta_y * rgrad)
                        param.copy_(new_param)

        params = [ManifoldParameter(p.detach().clone(), manifold=mfd) for mfd, p in zip(mfd_params, params)]

        true_hg = compute_hypergrad2(loss_lower, loss_upper, hparams, params, option='hinv', true_hessinv=true_hessinv,
                                    cg_iter=args.cg_iter, cg_gamma=args.cg_gamma)

        hypergrad = compute_hypergrad2(loss_lower, loss_upper, hparams, params,
                                    option=args.hygrad_opt, true_hessinv=true_hessinv,
                                    cg_iter=args.cg_iter, cg_gamma=args.cg_gamma)
        
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
        loss_u_all = [loss_upper(hparams, params).item()]
        hg_norm_all = [hgradnorm]
        runtime = [0]
        hg_error_all = [hg_error]
        print(f"Epoch {0}: "
            f"loss upper: {loss_u_all[-1]:.4f}, "
            f"hypergrad norm: {hgradnorm:.4e},"
            f"hg error: {hg_error_all[-1]:.4e}")
        
        # main run
        for ep in range(1,args.epoch+1):

            hparams, params, loss_u, hgradnorm, step_time, hg_error = RHGDstep(loss_lower, loss_upper, hparams, params, args,
                                                                    data=None, true_hessinv=true_hessinv)

            loss_u_all.append(loss_u)
            runtime.append(step_time)
            hg_error_all.append(hg_error)
            hg_norm_all.append(hgradnorm)
            epochs_all.append(ep)

            print(f"Epoch {ep}: "
                f"loss upper: {loss_u:.4f}, "
                f"hypergrad norm: {hgradnorm:.4e},"
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

    filename = res_folder + 'simple_n' + str(n) + 'd' + str(d) + 'r' + str(r) + '_RHGD_lr' + str(args.eta_x) + '_loweriter' + str(args.lower_iter) + '.pickle'

    with open(filename, 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)