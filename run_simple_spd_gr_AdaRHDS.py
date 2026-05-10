######## the experiment for Synthetic problem (AdaRHD-S) ########
"""
Algorithm: Adaptive Riemannian Hypergradient Descent with Single-loop (AdaRHD-S)
This script is strictly structured based on the reference D-TFBO code.
"""

import argparse
import torch
import os
import sys
import time
import pickle
import random
import numpy as np
from datetime import datetime

from geoopt import ManifoldParameter, Stiefel
from src.AdaRHD_manifolds import SymmetricPositiveDefiniteMod
from geoopt.linalg import sym_invm, sym_inv_sqrtm2
from scipy.linalg import solve_continuous_lyapunov

# Import existing toolkits
from src.AdaRHD_utils import autograd, batch_egrad2rgrad, dot

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# ==========================================
# 1. Core Mathematical Functions
# ==========================================

def true_hessinv(loss_lower, hparams, params, data_lower, tangents, X, Y, lam, d):
    W = hparams[0]
    M = params[0]
    Minvhalf, Mhalf = sym_inv_sqrtm2(M)
    G = (tangents[0].T + tangents[0]) 
    
    A = (X.T @ X)
    B = (W @ (Y.T @ Y) @ W.T) + lam * torch.eye(d, device=W.device)
    
    lhs = Mhalf @ A @ Mhalf + Minvhalf @ B @ Minvhalf
    lhs = 0.5 * (lhs + lhs.T)
    rhs = Minvhalf @ G @ Minvhalf
    
    U = solve_continuous_lyapunov(lhs.cpu().detach().numpy(), rhs.cpu().detach().numpy())
    U = torch.from_numpy(U).float().to(A.device)
    U = (U + U.transpose(-1, -2)) / 2
    U = Mhalf @ U @ Mhalf
    return [U]

def loss_lower(hparams, params, X, Y, lam):
    W = hparams[0]
    M = params[0]
    Minv = sym_invm(M)
    loss = (M * (X.T @ X)).sum() + (Minv * (W @ (Y.T @ Y) @ W.T)).sum() + lam * torch.trace(Minv)
    return loss

def loss_upper(hparams, params, X, Y):
    W = hparams[0]
    M = params[0]
    loss = -torch.trace(M @ X.T @ Y @ W.T)
    return loss

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
    parser.add_argument('--alpha', type=float, default=2)
    parser.add_argument('--beta', type=float, default=2)
    parser.add_argument('--gamma', type=float, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=50000)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--d', type=int, default=50)
    parser.add_argument('--r', type=int, default=20)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--log', type=bool, default=False)
    args = parser.parse_args()

    alpha0 = args.alpha

    if args.log:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_filename = f"simple_spd_gr_AdaRHD_S_log_{timestamp}.txt"
        sys.stdout = Logger(log_filename)

    np.random.seed(args.seed)
    seeds = np.random.randint(0, 10000, size=args.runs).tolist()

    for run in range(args.runs):
        seed = seeds[run]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

        n, d, r, lam = args.n, args.d, args.r, 0.01

        # Data Generation
        X = torch.randn(n, d, device=device); X = X / torch.norm(X)
        Y = torch.randn(n, r, device=device); Y = Y / torch.norm(Y)

        # Manifold Parameters Initialization
        spd = SymmetricPositiveDefiniteMod()
        st = Stiefel(canonical=False)
        hparams = [ManifoldParameter(st.random(d, r, device=device), manifold=st)]
        params = [ManifoldParameter(torch.eye(d, device=device), manifold=spd)]
        
        # Single-loop State: Initialize and maintain adjoint variable v
        v = [torch.zeros_like(params[0].data, device=device)]

        # Adaptive Step Accumulators
        a_sq, b_sq, c_sq = args.alpha**2, args.beta**2, args.gamma**2
        
        epochs_all, loss_u_all, hg_norm_all, hg_error_all, runtime_all = [], [], [], [], []

        print(f"Starting Run {run+1} with S-TFBO (Single-loop) logic...")

        for ep in range(args.epoch):
            start_time = time.time()

            # ---------------------------------------------------------
            # 1. Gradient Computations (Single-loop Logic)
            # ---------------------------------------------------------
            
            # Lower Gradient
            l_l = loss_lower(hparams, params, X, Y, lam)
            grad_y_g = torch.autograd.grad(l_l, params, create_graph=True)
            rgrad_y_g = batch_egrad2rgrad(params, grad_y_g)

            # Upper Gradient
            l_u = loss_upper(hparams, params, X, Y)
            grad_y_f = torch.autograd.grad(l_u, params, retain_graph=True)
            rgrad_y_f = batch_egrad2rgrad(params, grad_y_f)

            # Adjoint Update Term (HVP: nabla_yy g * v)
            gv_dot = dot(grad_y_g, v)
            hvp_y = torch.autograd.grad(gv_dot, params, retain_graph=True)
            rhvp_y = batch_egrad2rgrad(params, hvp_y)
            grad_v_R = [rh - rf for rh, rf in zip(rhvp_y, rgrad_y_f)]

            # Hypergradient Term (nabla_x f - nabla_xy g * v)
            grad_x_f = torch.autograd.grad(l_u, hparams, retain_graph=True)
            rgrad_x_f = batch_egrad2rgrad(hparams, grad_x_f)
            hvp_x = torch.autograd.grad(gv_dot, hparams, retain_graph=True)
            rhvp_x = batch_egrad2rgrad(hparams, hvp_x)
            G_hyper = [rf - rx for rf, rx in zip(rgrad_x_f, rhvp_x)]

            # ---------------------------------------------------------
            # 2. Hypergradient Error Calculation (vs. Analytical Solution)
            # ---------------------------------------------------------
            hg_error_val = 0.0
            with torch.no_grad():
                v_true = true_hessinv(None, hparams, params, None, rgrad_y_f, X, Y, lam, d)
            
            # Use temporary graph for hvp_x_true to avoid interfering with main flow
            l_l_tmp = loss_lower(hparams, params, X, Y, lam)
            grad_y_g_tmp = torch.autograd.grad(l_l_tmp, params, create_graph=True)
            gv_dot_true = dot(grad_y_g_tmp, v_true)
            rhvp_x_true = batch_egrad2rgrad(hparams, torch.autograd.grad(gv_dot_true, hparams))
            
            G_true = [rf.detach() - rx for rf, rx in zip(rgrad_x_f, rhvp_x_true)]
            
            with torch.no_grad():
                diff = [gh - gt for gh, gt in zip(G_hyper, G_true)]
                err_sq = hparams[0].manifold.inner(hparams[0], diff[0]).item()
                hg_error_val = np.sqrt(max(err_sq, 0))

            # ---------------------------------------------------------
            # 3. Adaptive Updates and Manifold Projections
            # ---------------------------------------------------------
            with torch.no_grad():
                # Accumulate step norms
                norm_gy = params[0].manifold.inner(params[0], rgrad_y_g[0]).item()
                b_sq += norm_gy
                bt = np.sqrt(b_sq)

                norm_gvR = params[0].manifold.inner(params[0], grad_v_R[0]).item()
                c_sq += norm_gvR
                ct = np.sqrt(c_sq)

                norm_gh = hparams[0].manifold.inner(hparams[0], G_hyper[0]).item()
                current_hgrad_norm = np.sqrt(max(norm_gh, 0))
                a_sq += norm_gh
                at = np.sqrt(a_sq)

                dt = max(bt, ct)

                # Parameter Updates (Retractions)
                # y update
                new_y = params[0].manifold.retr(params[0], -1.0 / bt * rgrad_y_g[0])
                params[0].copy_(new_y)

                # v update (linear space update)
                v[0] = v[0] - 1.0 / dt * grad_v_R[0]

                # x update
                new_x = hparams[0].manifold.retr(hparams[0], -1.0 / (at * dt) * G_hyper[0])
                hparams[0].copy_(new_x)

            step_time = time.time() - start_time
            
            # Storage
            loss_u_all.append(l_u.item())
            hg_norm_all.append(current_hgrad_norm)
            hg_error_all.append(hg_error_val)
            runtime_all.append(step_time)
            epochs_all.append(ep)

            if ep % 10 == 0:
                print(f"Epoch {ep}: loss_u={l_u.item():.4f}, hg_norm={current_hgrad_norm:.4e}, hg_error={hg_error_val:.4e}, time={step_time:.3f}")

            # Convergence Check
            if current_hgrad_norm < args.tol:
                print(f"AdaRHD converged at epoch {ep} with hypergrad norm {current_hgrad_norm:.16e}. ")
                break

        # Save Results
        res_folder = './results/'
        if not os.path.exists(res_folder): os.makedirs(res_folder)
        filename = f"{res_folder}simple_n{n}d{d}r{r}_AdaRHD_S_seed{seed}.pickle"
        
        stats = {
            'epochs': epochs_all,
            'loss_upper': loss_u_all,
            'hg_norm': hg_norm_all,
            'hg_error': hg_error_all,
            'runtime': runtime_all
        }
        with open(filename, 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Experiment Completed.")