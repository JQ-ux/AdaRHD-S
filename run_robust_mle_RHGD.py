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

from geoopt import ManifoldParameter
from src.AdaRHD_manifolds import EuclideanSimplexMod, SymmetricPositiveDefiniteMod
from geoopt.linalg import sym_invm, sym_inv_sqrtm2
from scipy.linalg import solve_continuous_lyapunov
import numpy as np
from rhgd.utils import autograd, compute_hypergrad2
from rhgd.optimizer import RHGDstep
import pickle
import random

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def mle(S, x):
    # S: torch.Tensor (d, d), x: torch.Tensor (d,)
    return 0.5 * (torch.logdet(S) + x @ torch.linalg.solve(S, x))

def loss_lower(hparams, params, data=None):
    S = params[0]
    y = hparams[0]
    # data: numpy array (n, d), convert to torch if needed
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).to(S.device).type(S.dtype)
    return sum([y[i] * mle(S, data[i]) for i in range(len(y))])

def loss_upper(hparams, params, data=None, reg=1):
    S = params[0]
    y = hparams[0]
    n = len(y)
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).to(S.device).type(S.dtype)
    return sum([- y[i] * mle(S, data[i]) for i in range(n)]) + \
           reg * torch.norm(y - 1 / n) ** 2

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
    parser.add_argument('--eta_x', type=float, default=0.1)
    parser.add_argument('--eta_y', type=float, default=0.1)
    parser.add_argument('--lower_iter', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=305)
    parser.add_argument('--hygrad_opt', type=str, default='cg', choices=['hinv', 'cg', 'ns', 'ad'])
    parser.add_argument('--cg_gamma', type=float, default=0.)
    parser.add_argument('--cg_iter', type=int, default=50)
    parser.add_argument('--ns_gamma', type=float, default=0.1)
    parser.add_argument('--ns_iter', type=int, default=50)
    parser.add_argument('--compute_hg_error', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--d', type=int, default=50)
    parser.add_argument('--log', type=bool, default=True)
    args = parser.parse_args()

    if args.log:
        # Create a log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        log_filename = f"robust_mle_RHGD_{args.lower_iter}_n{args.n}_d{args.d}_log_{timestamp}.txt"
        sys.stdout = Logger(log_filename)

    # Generate random seeds for multiple runs
    np.random.seed(args.seed)
    seeds = np.random.randint(0, 10000, size=args.runs).tolist()
    print(f"Using seeds: {seeds}")
    print(f"n: {args.n}, d: {args.d}")

    runtime_all_runs = []
    loss_u_all_runs = []
    total_hgradnorm_all_runs = []

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

        y = 1/n * torch.ones(n, device=device, dtype=torch.float32)
        eu = EuclideanSimplexMod(ndim=1)
        spd = SymmetricPositiveDefiniteMod()
        S = spd.random((d, d), device=device, dtype=torch.float32)
        mvn = torch.distributions.MultivariateNormal(torch.zeros(d), S)
        data = [mvn.sample((n,))]

        hparams = [ManifoldParameter(y, manifold=eu)]
        params = [ManifoldParameter(S, manifold=spd)]
        mfd_params = [spd]

        # initial run
        for ii in range(args.lower_iter):
            if args.hygrad_opt == 'ad':
                grad = autograd(loss_lower(hparams, params, data[0]), params, create_graph=True)
                rgrad = [mfd.egrad2rgrad(param, egrad) for mfd, egrad, param in zip(mfd_params, grad, params)]
                params = [mfd.retr(param, - args.eta_y * rg) for mfd, param, rg in zip(mfd_params, params, rgrad)]
            else:
                grad = autograd(loss_lower(hparams, params, data[0]), params)
                with torch.no_grad():
                    for param, egrad in zip(params, grad):
                        rgrad = param.manifold.egrad2rgrad(param, egrad)
                        new_param = param.manifold.retr(param, -args.eta_y * rgrad)
                        param.copy_(new_param)

        params = [ManifoldParameter(p.detach().clone(), manifold=mfd) for mfd, p in zip(mfd_params, params)]

        hypergrad = compute_hypergrad2(loss_lower, loss_upper, hparams, params, data_lower=data[0], data_upper=data[0],
                                    option=args.hygrad_opt, true_hessinv=None,
                                    cg_iter=args.cg_iter, cg_gamma=args.cg_gamma)
        with torch.no_grad():
            hgradnorm = 0
            for hparam, hg in zip(hparams, hypergrad):
                hgradnorm += hparam.manifold.inner(hparam, hg).item() / len(hparams)

        loss_u_all = [loss_upper(hparams, params, data[0]).item()]
        total_hgradnorm_all = [hgradnorm * len(hparams)]
        runtime = [0]
        print(f"Epoch {0}: "
            f"loss upper: {loss_u_all[-1]:.4f}, "
            f"hypergrad norm: {hgradnorm:.4e}")
        
        # main run
        for ep in range(1, args.epoch+1):

            hparams, params, loss_u, hgradnorm, step_time, hg_error = RHGDstep(loss_lower, loss_upper, hparams, params, args,
                                                                    data=data, true_hessinv=None)

            loss_u_all.append(loss_u)
            runtime.append(step_time)
            total_hgradnorm = hgradnorm * len(hparams)
            total_hgradnorm_all.append(total_hgradnorm)

            print(f"Epoch {ep}: "
                f"loss upper: {loss_u:.4f}, "
                f"hypergrad norm: {hgradnorm:.4e}, "
                f"step time: {step_time:.4f}")
        
        runtime_all_runs.append(torch.tensor(runtime).unsqueeze(0))
        loss_u_all_runs.append(torch.tensor(loss_u_all).unsqueeze(0))
        total_hgradnorm_all_runs.append(torch.tensor(total_hgradnorm_all).unsqueeze(0))
    
        stats = {'runtime': runtime_all_runs, 
                "loss_upper": loss_u_all_runs, 
                'total_hgradnorm': total_hgradnorm_all_runs}

        res_folder = './results/'
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)

        filename = res_folder + 'robust_mle_n' + str(n) + 'd' + str(d) + '_RHGD_' + str(args.lower_iter) + '_lr' + str(args.eta_x) + '.pickle'

        with open(filename, 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)