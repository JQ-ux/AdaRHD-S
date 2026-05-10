######## experiment for Shallow SPDNet (AdaRHD-S) ########
"""
Algorithm: Adaptive Riemannian Hypergradient Descent with Single-loop (AdaRHD-S)
Target: Hyperparameter Representation Learning on Shallow SPDNet
"""

import torch
import geoopt
import numpy as np
import os
import sys
import time
import pickle
import random
import argparse
from datetime import datetime

from geoopt import linalg, ManifoldParameter
from geoopt import SymmetricPositiveDefinite, Stiefel
from src.AdaRHD_manifolds import EuclideanMod
from src.AdaRHD_utils import autograd, batch_egrad2rgrad, dot

# ==========================================
# 1. Core Mathematical Functions
# ==========================================

def vec(X):
    """Reshape a symmetric matrix into a vector by extracting its upper-triangular part"""
    d = X.shape[-1]
    return X[..., torch.triu_indices(d, d)[0], torch.triu_indices(d, d)[1]]

def shallowSPDnet(hparams, params, A):
    W = hparams[0]
    gamma = params[0]
    # Mapping: W^T * A * W
    Z = W.transpose(-1, -2) @ A @ W
    Z = linalg.sym_logm(Z)
    Z = vec(Z)
    return Z

def loss_lower(hparams, params, data, lam):
    data_X, data_y = data
    gamma = params[0]
    pred = shallowSPDnet(hparams, params, data_X)
    # Mean Squared Error + L2 Regularization
    loss = 0.5 * torch.norm(pred @ gamma - data_y) ** 2 / data_X.shape[0] + 0.5 * lam * torch.norm(gamma) ** 2
    return loss

def loss_upper(hparams, params, data):
    data_X, data_y = data
    gamma = params[0]
    pred = shallowSPDnet(hparams, params, data_X)
    loss = 0.5 * torch.norm(pred @ gamma - data_y) ** 2 / data_X.shape[0]
    return loss

def true_hessinv(loss_lower, hparams, params, data_lower, tangents, lam, device):
    data_X, data_y = data_lower
    predg = shallowSPDnet(hparams, params, data_X)
    # Exact Hessian for linear regression part
    ehessg = (predg.transpose(-1, -2) @ predg) / data_X.shape[0] + lam * torch.eye(predg.shape[1], device=device)
    hessinvgrad = [torch.linalg.solve(ehessg, tangents[0])]
    return hessinvgrad

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
    parser.add_argument('--alpha', type=float, default=2.0)
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=100000)
    parser.add_argument('--N', type=int, default=200)
    parser.add_argument('--d', type=int, default=50)
    parser.add_argument('--r', type=int, default=10)
    parser.add_argument('--tol', type=float, default=8e-1)
    parser.add_argument('--log', type=bool, default=False)
    args = parser.parse_args()

    # Dynamic Path Setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    res_folder = os.path.join(current_dir, "results")
    if not os.path.exists(res_folder): os.makedirs(res_folder)

    if args.log:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_filename = os.path.join(current_dir, f"hyrep_spd_AdaRHD_S_log_{timestamp}.txt")
        sys.stdout = Logger(log_filename)

    np.random.seed(args.seed)
    seeds = np.random.randint(0, 10000, size=args.runs).tolist()

    for run in range(args.runs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seed = seeds[run]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Manifold Setup
        mfd = SymmetricPositiveDefinite()
        stiefel = Stiefel(canonical=False)
        euclidean = EuclideanMod(ndim=1)
        
        N, d, r, lam = args.N, args.d, args.r, 0.1

        # Data Generation
        A = mfd.random(N, d, d, device=device)
        Wstar = stiefel.random(d, r, device=device)
        gammastar = torch.randn(int(r * (r + 1) / 2), device=device)

        y_full = shallowSPDnet([Wstar], [gammastar], A) @ gammastar + torch.randn(N, device=device)
        
        Atr, ytr = A[:int(N/2)], y_full[:int(N/2)]
        Aval, yval = A[int(N/2):], y_full[int(N/2):]
        data_lower = [Atr, ytr]
        data_upper = [Aval, yval]

        # Init Parameters
        params = [ManifoldParameter(euclidean.random(int(r * (r + 1) / 2), device=device), manifold=euclidean)]
        hparams = [ManifoldParameter(torch.eye(d, r, device=device), manifold=stiefel)]
        
        # Single-loop State: Adjoint variable v
        v = [torch.zeros_like(params[0].data, device=device)]

        # Adaptive Accumulators
        a_sq, b_sq, c_sq = args.alpha**2, args.beta**2, args.gamma**2
        
        epochs_all, loss_u_all, hg_norm_all, hg_error_all, runtime_all = [], [], [], [], []

        print(f"Starting Run {run+1} with S-TFBO (Single-loop) logic...")
        is_converged = False

        for ep in range(args.epoch):
            start_time = time.time()

            # ---------------------------------------------------------
            # 1. Gradient Computations (Single-loop)
            # ---------------------------------------------------------
            l_l = loss_lower(hparams, params, data_lower, lam)
            grad_y_g = torch.autograd.grad(l_l, params, create_graph=True)
            rgrad_y_g = batch_egrad2rgrad(params, grad_y_g)

            l_u = loss_upper(hparams, params, data_upper)
            grad_y_f = torch.autograd.grad(l_u, params, retain_graph=True)
            rgrad_y_f = batch_egrad2rgrad(params, grad_y_f)

            gv_dot = dot(grad_y_g, v)
            hvp_y = torch.autograd.grad(gv_dot, params, retain_graph=True)
            rhvp_y = batch_egrad2rgrad(params, hvp_y)
            grad_v_R = [rh - rf for rh, rf in zip(rhvp_y, rgrad_y_f)]

            grad_x_f = torch.autograd.grad(l_u, hparams, retain_graph=True)
            rgrad_x_f = batch_egrad2rgrad(hparams, grad_x_f)
            hvp_x = torch.autograd.grad(gv_dot, hparams, retain_graph=True)
            rhvp_x = batch_egrad2rgrad(hparams, hvp_x)
            G_hyper = [rf - rx for rf, rx in zip(rgrad_x_f, rhvp_x)]

            # ---------------------------------------------------------
            # 2. Hypergradient Error Calculation
            # ---------------------------------------------------------
            with torch.no_grad():
                v_true = true_hessinv(None, hparams, params, data_lower, rgrad_y_f, lam, device)
            
            l_l_tmp = loss_lower(hparams, params, data_lower, lam)
            grad_y_g_tmp = torch.autograd.grad(l_l_tmp, params, create_graph=True)
            gv_dot_true = dot(grad_y_g_tmp, v_true)
            rhvp_x_true = batch_egrad2rgrad(hparams, torch.autograd.grad(gv_dot_true, hparams))
            G_true = [rf.detach() - rx for rf, rx in zip(rgrad_x_f, rhvp_x_true)]
            
            with torch.no_grad():
                diff = [gh - gt for gh, gt in zip(G_hyper, G_true)]
                err_sq = hparams[0].manifold.inner(hparams[0], diff[0]).item()
                hg_error_val = np.sqrt(max(err_sq, 0))

            # ---------------------------------------------------------
            # 3. Adaptive Updates
            # ---------------------------------------------------------
            with torch.no_grad():
                norm_gy = params[0].manifold.inner(params[0], rgrad_y_g[0]).item()
                b_sq += norm_gy
                bt = np.sqrt(max(b_sq, 1e-8))

                norm_gvR = params[0].manifold.inner(params[0], grad_v_R[0]).item()
                c_sq += norm_gvR
                ct = np.sqrt(max(c_sq, 1e-8))

                norm_gh = hparams[0].manifold.inner(hparams[0], G_hyper[0]).item()
                current_hgrad_norm = np.sqrt(max(norm_gh, 0))
                a_sq += norm_gh
                at = np.sqrt(max(a_sq, 1e-8))

                dt = max(bt, ct)

                # Parameter Retractions
                params[0].copy_(params[0].manifold.retr(params[0], -1.0 / bt * rgrad_y_g[0]))
                v[0] = v[0] - 1.0 / dt * grad_v_R[0]
                hparams[0].copy_(hparams[0].manifold.retr(hparams[0], -1.0 / (at * dt) * G_hyper[0]))

            step_time = time.time() - start_time
            loss_u_all.append(l_u.item())
            hg_norm_all.append(current_hgrad_norm)
            hg_error_all.append(hg_error_val)
            runtime_all.append(step_time)
            epochs_all.append(ep)

            # Convergence Check
            if current_hgrad_norm < args.tol:
                print(f"AdaRHD converged at epoch {ep}: loss_u={l_u.item():.4f}, hg_norm={current_hgrad_norm:.4e}, hg_error={hg_error_val:.4e}")
                is_converged = True
                break

            if ep % 10 == 0:
                print(f"Epoch {ep}: loss_u={l_u.item():.4f}, hg_norm={current_hgrad_norm:.4e}, hg_error={hg_error_val:.4e}, time={step_time:.3f}")

        if not is_converged:
            print(f"Reached max epochs ({args.epoch}). Final hg_norm: {current_hgrad_norm:.4e}")

        # Storage matching the required format [run, epoch]
        epochs_all_runs = [torch.tensor(epochs_all).unsqueeze(0)]
        runtime_all_runs = [torch.tensor(runtime_all).unsqueeze(0)]
        loss_u_all_runs = [torch.tensor(loss_u_all).unsqueeze(0)]
        hg_error_all_runs = [torch.tensor(hg_error_all).unsqueeze(0)]
        hg_norm_all_runs = [torch.tensor(hg_norm_all).unsqueeze(0)]

        stats = {
            'epochs': epochs_all_runs, 
            'runtime': runtime_all_runs, 
            "loss_upper": loss_u_all_runs, 
            "hg_error": hg_error_all_runs, 
            'hg_norm': hg_norm_all_runs
        }
        
        filename = os.path.join(res_folder, f'shallow_hyrep_n{N}d{d}r{r}_AdaRHD_S_seed{seed}.pickle')
        with open(filename, 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Experiment Completed.")