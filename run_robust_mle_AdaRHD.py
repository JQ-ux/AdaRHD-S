######## the experiment for Synthetic problem ########

####  max_{W in St(d,r)} tr(M* X^T Y W^T),  
#### s.t. M* = argmin_{M in S_{++}^d} <M, X^T X> + <M^{-1}, W Y^T Y W^T + ν I>,

# structure of the code is derived from https://github.com/andyjm3/rhgd

import argparse
import torch

import os
import sys
from datetime import datetime

import argparse
import torch

from geoopt import ManifoldParameter
from src.AdaRHD_manifolds import EuclideanSimplexMod, SymmetricPositiveDefiniteMod
from geoopt.linalg import sym_invm, sym_inv_sqrtm2
from scipy.linalg import solve_continuous_lyapunov
import numpy as np
from src.AdaRHD_utils import autograd, compute_hypergrad_v
from src.AdaRHD_optimizer import AdaRHDstep
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

def loss_upper(hparams, params, data=None, reg=1e2):
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
    parser.add_argument('--alpha', type=float, default=10)
    parser.add_argument('--beta', type=float, default=10)
    parser.add_argument('--gamma', type=float, default=10)
    parser.add_argument('--y_subiter', type=int, default=50)
    parser.add_argument('--v_subiter', type=int, default=50)
    parser.add_argument('--hygrad_opt', type=str, default='cg', choices=['cg', 'gd', 'hinv'])
    parser.add_argument('--v_reg', type=float, default=0.)
    parser.add_argument('--compute_hg_error', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--d', type=int, default=50)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--log', type=bool, default=True)
    parser.add_argument('--epoch', type=int, default=300)
    args = parser.parse_args()
    alpha0 = args.alpha

    if args.log:
        # Create a log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        log_filename = f"robust_mle_AdaRHD_{args.hygrad_opt}_n{args.n}_d{args.d}_log_{timestamp}.txt"
        sys.stdout = Logger(log_filename)

    # Generate random seeds for multiple runs
    np.random.seed(args.seed)
    seeds = np.random.randint(0, 10000, size=args.runs).tolist()
    print(f"Using seeds: {seeds}")
    print(f"n: {args.n}, d: {args.d}")

    if args.hygrad_opt == 'gd':
        args.epoch = 500
        args.tol_y = args.tol_v = 1/args.epoch
    elif args.hygrad_opt == 'cg':
        args.epoch = 300
        args.tol_y = 1/args.epoch
        args.tol_v = 1e-10

    args.v = None
    
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
        for ii in range(args.y_subiter):
            if args.hygrad_opt == 'ad':
                grad = autograd(loss_lower(hparams, params, data[0]), params, create_graph=True)
                rgrad = [mfd.egrad2rgrad(param, egrad) for mfd, egrad, param in zip(mfd_params, grad, params)]
                params = [mfd.retr(param, - 1 / args.beta * rg) for mfd, param, rg in zip(mfd_params, params, rgrad)]
            else:
                grad = autograd(loss_lower(hparams, params, data[0]), params)
                with torch.no_grad():
                    for param, egrad in zip(params, grad):
                        rgrad = param.manifold.egrad2rgrad(param, egrad)
                        new_param = param.manifold.retr(param, -1 / args.beta * rgrad)
                        param.copy_(new_param)

        params = [ManifoldParameter(p.detach().clone(), manifold=mfd) for mfd, p in zip(mfd_params, params)]

        hypergrad, _, _ = compute_hypergrad_v(loss_lower, loss_upper, hparams, params, 
                                    data_lower=data[0], data_upper=data[0], v0=None, hygrad_opt=args.hygrad_opt, true_hessinv= None,
                                    iter = args.v_subiter, gd_gamma = args.gamma, v_reg=args.v_reg, tol=args.tol_v, verbose=args.verbose)
        
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
        total_hgradnorm0 = hgradnorm * len(hparams)
        total_hgradnorm = hgradnorm * len(hparams)
        args.alpha = np.sqrt(alpha0 ** 2 + total_hgradnorm)
        for ep in range(1, args.epoch+1):

            hparams, params, args.v, loss_u, hgradnorm, step_time, hg_error, inner_iter, hygrad_flag = AdaRHDstep(loss_lower, loss_upper, hparams, params, args,
                                                                        data=data, true_hessinv=None)
            
            if args.hygrad_opt == 'cg':
                args.v = None

            total_hgradnorm = hgradnorm * len(hparams)
            if ep == 1:
                total_hgradnorm0 = total_hgradnorm
            elif total_hgradnorm > 5 * total_hgradnorm0:
                hygrad_flag = False
             
            if hygrad_flag:
                args.alpha = np.sqrt(args.alpha ** 2 + total_hgradnorm)

            loss_u_all.append(loss_u)
            runtime.append(step_time)
            total_hgradnorm_all.append(total_hgradnorm)

            print(f"Epoch {ep}: "
                    f"loss upper: {loss_u:.4f}, "
                    f"hypergrad norm: {hgradnorm:.4e},"
                    f"alpha: {args.alpha:.4f}, "
                    f"step time: {step_time:.4f}")

            if ep > 305:
                break

            if ep % 5 == 0:
                args.y_subiter = min(args.y_subiter + 50, 500)
                if args.hygrad_opt == 'gd':
                    args.v_subiter = min(args.v_subiter + 50, 500)
                

        runtime_all_runs.append(torch.tensor(runtime).unsqueeze(0))
        loss_u_all_runs.append(torch.tensor(loss_u_all).unsqueeze(0))
        total_hgradnorm_all_runs.append(torch.tensor(total_hgradnorm_all).unsqueeze(0))

        stats = {'runtime': runtime_all_runs,
                'loss_upper': loss_u_all_runs, 'total_hgradnorm': total_hgradnorm_all_runs}

        res_folder = './results/'
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)

        filename = res_folder + 'robust_mle_n' + str(n) + 'd' + str(d) + '_AdaRHD_' + str(args.hygrad_opt) + '_lr' + str(1 / args.beta) + '.pickle'

        with open(filename, 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)