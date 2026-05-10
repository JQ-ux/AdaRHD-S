import torch
from torch import nn
import time
import numpy as np
from geoopt import linalg
import pickle
   
import os
import sys
from datetime import datetime

from torch.autograd import Function as F
from geoopt import linalg
import argparse

from geoopt import Stiefel, ManifoldParameter
from src.AdaRHD_manifolds import EuclideanMod
from scipy.io import loadmat

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def vec(X):
    """Reshape a symmetric matrix into a vector by extracting its upper-triangular part"""
    d = X.shape[-1]
    return X[..., torch.triu_indices(d, d)[0], torch.triu_indices(d, d)[1]]

class Re_op():
    """ Relu function and its derivative """
    _threshold = 1e-4

    @classmethod
    def fn(cls, S, param=None):
        return nn.Threshold(cls._threshold, cls._threshold)(S)

    @classmethod
    def fn_deriv(cls, S, param=None):
        return (S > cls._threshold).float()

def add_id_matrix(P, alpha):
    """Input P of shape (batch_size,1,n,n), Add Identity"""
    P = P + alpha * P.trace() * torch.eye(
        P.shape[-1], dtype=P.dtype, device=P.device)
    return P

def BatchDiag(P):
    """Input P: (batch_size,channels) vectors, Output Q: diagonal matrices"""
    batch_size, channels, n = P.shape
    Q = torch.zeros(batch_size, channels, n, n, dtype=P.dtype, device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            Q[i, j] = P[i, j].diag()
    return Q

def modeig_forward(P, op, eig_mode='svd', param=None):
    batch_size, channels, n, n = P.shape
    U = torch.zeros_like(P, device=P.device)
    S = torch.zeros(batch_size, channels, n, dtype=P.dtype, device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            if (eig_mode == 'svd'):
                # --- 多级鲁棒 SVD 逻辑 ---
                # 尝试不同的 epsilon 扰动，直到 SVD 收敛
                for eps in [1e-5, 1e-4, 1e-3, 1e-2]:
                    try:
                        input_mat = add_id_matrix(P[i, j], eps)
                        u, s, v = torch.svd(input_mat)
                        # 检查是否有 NaN，有些情况下 svd 不报错但出 NaN
                        if not torch.isnan(u).any():
                            U[i, j], S[i, j] = u, s
                            break
                    except RuntimeError:
                        if eps == 1e-2: 
                            raise RuntimeError(f"SVD failed even with eps=1e-2 at batch {i}, channel {j}")
                        continue 
    S_fn = op.fn(S, param)
    X = U.matmul(BatchDiag(S_fn)).matmul(U.transpose(2, 3))
    return X, U, S, S_fn

def modeig_backward(dx, U, S, S_fn, op, param=None):
    S_fn_deriv = BatchDiag(op.fn_deriv(S, param))
    SS = S[..., None].repeat(1, 1, 1, S.shape[-1])
    SS_fn = S_fn[..., None].repeat(1, 1, 1, S_fn.shape[-1])
    L = (SS_fn - SS_fn.transpose(2, 3)) / (SS - SS.transpose(2, 3))
    L[L == -np.inf] = 0
    L[L == np.inf] = 0
    L[torch.isnan(L)] = 0
    L = L + S_fn_deriv
    dp = L * (U.transpose(2, 3).matmul(dx).matmul(U))
    dp = U.matmul(dp).matmul(U.transpose(2, 3))
    return dp

class ReEig(F):
    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Re_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        U, S, S_fn = ctx.saved_tensors
        return modeig_backward(dx, U, S, S_fn, Re_op)

def SPDnet(hparams, params, A):
    W1, W2, W3 = hparams
    Z1 = W1.transpose(-1, -2) @ A @ W1
    A1 = ReEig.apply(Z1.unsqueeze(1))
    Z2 = W2.transpose(-1,-2) @ A1.squeeze(1) @ W2
    A2 = ReEig.apply(Z2.unsqueeze(1))
    Z3 = W3.transpose(-1,-2) @ A2.squeeze(1) @ W3
    Z = linalg.sym_logm(Z3)
    return vec(Z).squeeze()

def loss_lower_mfd(hparams, params, data, lam=0.1):
    data_X, data_y = data
    gamma = params[0]
    pred = SPDnet(hparams, params, data_X)
    loss = nn.functional.cross_entropy(pred @ gamma, data_y) + 0.5 * lam * torch.norm(gamma) ** 2
    return loss

def loss_upper_mfd(hparams, params, data):
    data_X, data_y = data
    gamma = params[0]
    pred = SPDnet(hparams, params, data_X)
    loss = nn.functional.cross_entropy(pred @ gamma, data_y)
    return loss

def compute_acc(network, hparams, params, data):
    data_X, data_y = data
    gamma = params[0]
    pred = network(hparams, params, data_X)
    logit = pred @ gamma
    y_pred = torch.argmax(logit, dim=1)
    acc = (y_pred == data_y).sum().float() / data_y.shape[0]
    return acc

def load_data_from_directory(directory, data_list, labels_list):
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):
            try:
                label_num = int(label)
                for file_name in os.listdir(label_path):
                    if file_name.endswith('.mat'):
                        file_path = os.path.join(label_path, file_name)
                        mat_data = loadmat(file_path)
                        data_key = [k for k in mat_data.keys() if not k.startswith('__')][0]
                        data_list.append(mat_data[data_key])
                        labels_list.append(label_num)
            except ValueError: continue

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=1.0, help='initial high-level step size')
    parser.add_argument('--beta', type=float, default=1.0, help='initial low-level step size')
    parser.add_argument('--gamma_step', type=float, default=1.0, help='initial hyper-auxiliary step size')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=3000)
    parser.add_argument('--log', type=bool, default=False)
    args = parser.parse_args()

    if args.log:
        timestamp = datetime.now().strftime("%Y%m%d")
        sys.stdout = Logger(f"deep_hyrep_spd_AdaRHD_S_log_{timestamp}.txt")

    train_path, val_path = 'data/spdface_400_inter_histeq/train', 'data/spdface_400_inter_histeq/val'
    data_X, y, data_X_test, test_y = [], [], [], []
    load_data_from_directory(train_path, data_X, y)
    load_data_from_directory(val_path, data_X_test, test_y)

    data_X = torch.from_numpy(np.array(data_X)).float()
    data_y = torch.from_numpy(np.array(y)-1).long()
    data_X_test = torch.from_numpy(np.array(data_X_test)).float()
    data_y_test = torch.from_numpy(np.array(test_y)-1).long()

    labels_to_keep = [0, 1, 2, 3, 4, 5, 6]
    train_mask = torch.tensor([label.item() in labels_to_keep for label in data_y])
    test_mask = torch.tensor([label.item() in labels_to_keep for label in data_y_test])
    data_X, data_y = data_X[train_mask], data_y[train_mask]
    data_X_test, data_y_test = data_X_test[test_mask], data_y_test[test_mask]
    
    label_mapping = {original: idx for idx, original in enumerate(labels_to_keep)}
    data_y = torch.tensor([label_mapping[l.item()] for l in data_y])
    data_y_test = torch.tensor([label_mapping[l.item()] for l in data_y_test])

    np.random.seed(args.seed)
    seeds = np.random.randint(0, 10000, size=args.runs).tolist()

    for data_ratio in [0.05]:
        print(f"\n--- Training with data_ratio: {data_ratio} ---")
        acc_all_runs, test_acc_all_runs, loss_all_runs, runtime_all_runs = [], [], [], []

        for run in range(args.runs):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # --- START OF MODIFICATION: SEEDING ---
            current_seed = seeds[run]
            torch.manual_seed(current_seed)
            np.random.seed(current_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(current_seed)
            # --- END OF MODIFICATION ---
            
            # Data Splitting
            train_X_list, val_X_list, train_y_list, val_y_list = [], [], [], []
            for c in range(len(labels_to_keep)):
                sub = data_X[data_y == c]
                idx = torch.randperm(sub.shape[0])
                n_tr = int(len(idx) * data_ratio)
                train_X_list.append(sub[idx[:n_tr]])
                val_X_list.append(sub[idx[n_tr:2*n_tr]])
                train_y_list.extend([c]*n_tr)
                val_y_list.extend([c]*n_tr)

            train_X, train_y = torch.cat(train_X_list).to(device), torch.tensor(train_y_list).to(device)
            val_X, val_y = torch.cat(val_X_list).to(device), torch.tensor(val_y_list).to(device)
            test_X, test_y = data_X_test.to(device), data_y_test.to(device)

            # Model Initialization
            stiefel, euclidean = Stiefel(canonical=False), EuclideanMod(ndim=2)
            hparams = [ManifoldParameter(stiefel.random(400, 100, device=device), manifold=stiefel),
                       ManifoldParameter(stiefel.random(100, 20, device=device), manifold=stiefel),
                       ManifoldParameter(stiefel.random(20, 5, device=device), manifold=stiefel)]
            params = [ManifoldParameter(euclidean.random(15, len(labels_to_keep), device=device), manifold=euclidean)]
            
            # Hyper-auxiliary variable v
            v = [torch.zeros_like(p, device=device) for p in params]
            
            # AdaRHD-S Accumulators (Algorithm 1)
            at_sq, bt_sq, ct_sq = args.alpha**2, args.beta**2, args.gamma_step**2
            
            acc_history, test_acc_history, loss_history, time_history = [], [], [], []
            start_time = time.time()

            print(f"Run {run+1}/{args.runs} started on {device}")

            for ep in range(1, args.epoch + 1):
                ep_start = time.time()
                
                # 1. Compute Lower Gradient (Gyg)
                loss_l = loss_lower_mfd(hparams, params, [train_X, train_y])
                grad_y = torch.autograd.grad(loss_l, params, create_graph=True)
                
                # 2. Compute Upper Gradient for BF (Gbf) and auxiliary gradient
                loss_u = loss_upper_mfd(hparams, params, [val_X, val_y])
                
                # V-update components
                grad_u_y = torch.autograd.grad(loss_u, params, retain_graph=True)
                
                # Hessian-vector product
                v_grad_y = sum([(v_i * g_i).sum() for v_i, g_i in zip(v, grad_y)])
                hvp_y = torch.autograd.grad(v_grad_y, params, retain_graph=True)
                
                grad_v_R = [(gi + hi - gui) for gi, hi, gui in zip(grad_y, hvp_y, grad_u_y)]
                
                # Hypergradient Gbf approximation
                grad_u_x = torch.autograd.grad(loss_u, hparams, retain_graph=True)
                v_grad_y_for_x = sum([(v_i * g_i).sum() for v_i, g_i in zip(v, grad_y)])
                hvp_x = torch.autograd.grad(v_grad_y_for_x, hparams)
                
                grad_x_f = [(gux - hx) for gux, hx in zip(grad_u_x, hvp_x)]

                # 3. Update Adaptive Step Sizes
                bt_sq += sum([g.norm()**2 for g in grad_y]).item()
                ct_sq += sum([g.norm()**2 for g in grad_v_R]).item()
                dt_val = max(np.sqrt(bt_sq), np.sqrt(ct_sq))
                at_sq += sum([g.norm()**2 for g in grad_x_f]).item()
                
                # 4. Update Parameters
                with torch.no_grad():
                    for p, g in zip(params, grad_y):
                        new_p = p.manifold.retr(p, - (1.0 / np.sqrt(bt_sq)) * g)
                        p.copy_(new_p)
                    
                    for vi, gv in zip(v, grad_v_R):
                        vi.sub_(gv, alpha=1.0 / dt_val)
                    
                    for hp, gx in zip(hparams, grad_x_f):
                        new_hp = hp.manifold.retr(hp, - (1.0 / (np.sqrt(at_sq) * dt_val)) * gx)
                        hp.copy_(new_hp)

                # Recording
                step_duration = time.time() - ep_start
                val_acc = compute_acc(SPDnet, hparams, params, [val_X, val_y])
                test_acc = compute_acc(SPDnet, hparams, params, [test_X, test_y])
                
                acc_history.append(val_acc.item())
                test_acc_history.append(test_acc.item())
                loss_history.append(loss_u.item())
                time_history.append(step_duration)

                if ep % 50 == 0 or ep == 1:
                    print(f"Ep {ep:03d} | LossU: {loss_u:.4e} | ValAcc: {val_acc*100:.2f}% | TestAcc: {test_acc*100:.2f}% | Time: {step_duration:.2f}s")

            acc_all_runs.append(torch.tensor(acc_history).unsqueeze(0))
            test_acc_all_runs.append(torch.tensor(test_acc_history).unsqueeze(0))
            loss_all_runs.append(torch.tensor(loss_history).unsqueeze(0))
            runtime_all_runs.append(torch.tensor(time_history).unsqueeze(0))

        # Save Results
        res_folder = './results/'
        os.makedirs(res_folder, exist_ok=True)
        stats = {'runtime': runtime_all_runs, 'loss_upper': loss_all_runs, 'accuracy': acc_all_runs, 'test_accuracy': test_acc_all_runs}
        filename = f"{res_folder}hyrep_spd_AdaRHD_S_lr{1/args.beta}_data_ratio{data_ratio}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Results saved to {filename}")