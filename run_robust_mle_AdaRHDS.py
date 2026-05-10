######## the experiment for Synthetic problem (Single-loop AdaRHD-S) ########

import argparse
import torch
import os
import sys
import time
import pickle
import random
import numpy as np
from datetime import datetime

from geoopt import ManifoldParameter
from src.AdaRHD_manifolds import EuclideanSimplexMod, SymmetricPositiveDefiniteMod
from src.AdaRHD_utils import autograd, batch_egrad2rgrad, dot

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# ==========================================
# 1. Core Mathematical Functions
# ==========================================

def mle(S, x):
    # S: torch.Tensor (d, d), x: torch.Tensor (d,)
    return 0.5 * (torch.logdet(S) + x @ torch.linalg.solve(S, x))

def loss_lower(hparams, params, data=None):
    S = params[0]
    y = hparams[0]
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
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ==========================================
# 2. Main Entry
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=10)
    parser.add_argument('--beta', type=float, default=10)
    parser.add_argument('--gamma', type=float, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--d', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=1000)  
    parser.add_argument('--log', type=bool, default=True)
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()

    if args.log:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_filename = f"robust_mle_AdaRHD_S_n{args.n}_d{args.d}_log_{timestamp}.txt"
        sys.stdout = Logger(log_filename)

    np.random.seed(args.seed)
    seeds = np.random.randint(0, 10000, size=args.runs).tolist()
    print(f"Using seeds: {seeds}")
    print(f"n: {args.n}, d: {args.d}")

    # Results containers (strictly keeping your list format)
    runtime_all_runs = []
    loss_u_all_runs = []
    total_hgradnorm_all_runs = []

    for run in range(args.runs):
        seed = seeds[run]
        print(f"Run {run+1}/{args.runs} with seed {seed}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Deterministic setup
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        # Manifolds initialization
        eu = EuclideanSimplexMod(ndim=1)
        spd = SymmetricPositiveDefiniteMod()
        
        # Data generation
        S_true = spd.random((args.d, args.d), device=device)
        mvn = torch.distributions.MultivariateNormal(torch.zeros(args.d, device=device), S_true)
        data_samples = mvn.sample((args.n,))
        
        # Initial Parameters
        hparams = [ManifoldParameter(1.0/args.n * torch.ones(args.n, device=device), manifold=eu)]
        params = [ManifoldParameter(spd.random((args.d, args.d), device=device), manifold=spd)]
        
        # Single-loop State Variables
        v = [torch.zeros_like(params[0].data, device=device)]
        a_sq, b_sq, c_sq = args.alpha**2, args.beta**2, args.gamma**2
        
        # Iteration-level storage
        runtime, loss_u_curr, hgradnorm_curr = [], [], []

        print(f"Starting Run {run+1} with Single-loop AdaRHD-S...")

        for ep in range(args.epoch + 1):
            start_time = time.time()

            # --- 1. Compute Gradients ---
            # Lower gradient: G_y g
            l_l = loss_lower(hparams, params, data_samples)
            grad_y_g = torch.autograd.grad(l_l, params, create_graph=True)
            rgrad_y_g = batch_egrad2rgrad(params, grad_y_g)

            # Adjoint logic: grad_v R
            l_u = loss_upper(hparams, params, data_samples)
            grad_y_f = torch.autograd.grad(l_u, params, retain_graph=True)
            rgrad_y_f = batch_egrad2rgrad(params, grad_y_f)
            
            gv_dot = dot(grad_y_g, v)
            hvp_y = torch.autograd.grad(gv_dot, params, retain_graph=True)
            rhvp_y = batch_egrad2rgrad(params, hvp_y)
            grad_v_R = [rh - rf for rh, rf in zip(rhvp_y, rgrad_y_f)]

            # Hypergradient: G_b f
            grad_x_f = torch.autograd.grad(l_u, hparams, retain_graph=True)
            rgrad_x_f = batch_egrad2rgrad(hparams, grad_x_f)
            hvp_x = torch.autograd.grad(gv_dot, hparams, retain_graph=True)
            rhvp_x = batch_egrad2rgrad(hparams, hvp_x)
            G_hyper = [rf - rx for rf, rx in zip(rgrad_x_f, rhvp_x)]

            # --- 2. Adaptive Updates (Algorithm 1) ---
            with torch.no_grad():
                # Accumulate Squares
                norm_gy_sq = params[0].manifold.inner(params[0], rgrad_y_g[0]).item()
                b_sq += norm_gy_sq
                bt = np.sqrt(max(b_sq, 1e-8))

                norm_gvR_sq = params[0].manifold.inner(params[0], grad_v_R[0]).item()
                c_sq += norm_gvR_sq
                ct = np.sqrt(max(c_sq, 1e-8))
                
                dt = max(bt, ct)

                norm_gh_sq = hparams[0].manifold.inner(hparams[0], G_hyper[0]).item()
                current_hgrad_norm = np.sqrt(max(norm_gh_sq, 0))
                a_sq += norm_gh_sq
                at = np.sqrt(max(a_sq, 1e-8))

                # Retractions
                # Line 7: y_{t+1}
                params[0].copy_(params[0].manifold.retr(params[0], -1.0 / bt * rgrad_y_g[0]))
                # Line 8: v_{t+1}
                v[0] = v[0] - 1.0 / dt * grad_v_R[0]
                # Line 9: x_{t+1}
                hparams[0].copy_(hparams[0].manifold.retr(hparams[0], -1.0 / (at * dt) * G_hyper[0]))

            step_time = time.time() - start_time
            
            # Storage (strictly following previous conventions)
            loss_u_curr.append(l_u.item())
            runtime.append(step_time if ep > 0 else 0)
            hgradnorm_curr.append(current_hgrad_norm)

            if ep % 50 == 0:
                print(f"Epoch {ep}: loss_u={l_u.item():.4f}, hg_norm={current_hgrad_norm:.4e}, time={step_time:.4f}")

        # Wrap in list/tensor for multi-run format [run, epoch]
        runtime_all_runs.append(torch.tensor(runtime).unsqueeze(0))
        loss_u_all_runs.append(torch.tensor(loss_u_curr).unsqueeze(0))
        total_hgradnorm_all_runs.append(torch.tensor(hgradnorm_curr).unsqueeze(0))

    # --- Final Stats and Pickle Save ---
    stats = {
        'runtime': runtime_all_runs,
        'loss_upper': loss_u_all_runs,
        'total_hgradnorm': total_hgradnorm_all_runs
    }

    res_folder = './results/'
    if not os.path.exists(res_folder): os.makedirs(res_folder)

    filename = os.path.join(res_folder, f'robust_mle_n{args.n}_d{args.d}_AdaRHD_S_lr{1.0/args.beta}.pickle')
    with open(filename, 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Experiment Completed. File saved to {filename}")