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
from src.AdaRHD_optimizer import AdaRHDstep
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
    '''
    Input P of shape (batch_size,1,n,n)
    Add Id
    '''
    P = P + alpha * P.trace() * torch.eye(
        P.shape[-1], dtype=P.dtype, device=P.device)
    return P

def modeig_forward(P, op, eig_mode='svd', param=None):
    '''
    Generic forward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    batch_size, channels, n, n = P.shape
    U, S = torch.zeros_like(P, device=P.device), torch.zeros(batch_size,
                                                       channels,
                                                       n,
                                                       dtype=P.dtype,
                                                       device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            if (eig_mode == 'eig'):
                s, U[i, j] = torch.eig(P[i, j], True)
                S[i, j] = s[:, 0]
            elif (eig_mode == 'svd'):
                U[i, j], S[i, j], _ = torch.svd(add_id_matrix(P[i, j], 1e-5))
    S_fn = op.fn(S, param)
    X = U.matmul(BatchDiag(S_fn)).matmul(U.transpose(2, 3))
    return X, U, S, S_fn


def BatchDiag(P):
    """
    Input P: (batch_size,channels) vectors of size (n)
    Output Q: (batch_size,channels) diagonal matrices of size (n,n)
    """
    batch_size, channels, n = P.shape  #batch size,channel depth,dimension
    Q = torch.zeros(batch_size, channels, n, n, dtype=P.dtype, device=P.device)
    for i in range(batch_size):  #can vectorize
        for j in range(channels):  #can vectorize
            Q[i, j] = P[i, j].diag()
    return Q

def modeig_backward(dx, U, S, S_fn, op, param=None):
    '''
    Generic backward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''

    #print("Correct back prop")
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
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Re_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Re_op)

def SPDnet(hparams, params, A):
    # A needs to be [batch, d, d]
    W1 = hparams[0]
    W2 = hparams[1]
    W3 = hparams[2]
    # 1st layer
    Z1 = W1.transpose(-1, -2) @ A @ W1
    A1 = ReEig.apply(Z1.unsqueeze(1))
    # 2nd layer
    Z2 = W2.transpose(-1,-2) @ A1.squeeze(1) @ W2
    A2 = ReEig.apply(Z2.unsqueeze(1))
    # 3rd layer
    Z3 = W3.transpose(-1,-2) @ A2.squeeze(1) @ W3
    Z = linalg.sym_logm(Z3)
    Z = vec(Z).squeeze()
    return Z

def loss_lower_mfd(hparams, params, data):
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
    acc = (y_pred == data_y).sum() / data_y.shape[0]
    return acc

# Function to load data from a directory
def load_data_from_directory(directory, data_list, labels_list):
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):
            try:
                label_num = int(label)  # Convert folder name to integer label
                print(f"Processing label: {label_num}")
                
                for file_name in os.listdir(label_path):
                    if file_name.endswith('.mat'):
                        file_path = os.path.join(label_path, file_name)
                        try:
                            mat_data = loadmat(file_path)
                            # Assuming the actual data is in the first non-standard key
                            data_key = [k for k in mat_data.keys() if not k.startswith('__')][0]
                            data_list.append(mat_data[data_key])
                            labels_list.append(label_num)
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
            except ValueError:
                print(f"Skipping non-numeric folder: {label}")

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
    parser.add_argument('--compute_hg_error', type=bool, default=True)
    parser.add_argument('--alpha', type=float, default=10)
    parser.add_argument('--beta', type=float, default=10)
    parser.add_argument('--gamma', type=float, default=10)
    parser.add_argument('--y_subiter', type=int, default=100)
    parser.add_argument('--v_subiter', type=int, default=50)
    parser.add_argument('--hygrad_opt', type=str, default='cg', choices=['cg', 'gd', 'hinv'])
    parser.add_argument('--v_reg', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--log', type=bool, default=False)
    args = parser.parse_args()
    alpha0 = args.alpha

    if args.log:
        # Create a log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        log_filename = f"deep_hyrep_spd_AdaRHD_{args.hygrad_opt}_log_{timestamp}.txt"
        sys.stdout = Logger(log_filename)

    if args.hygrad_opt == 'gd':
        args.epoch = 500
        args.tol_y = args.tol_v = 1/args.epoch
    elif args.hygrad_opt == 'cg':
        args.epoch = 300
        args.tol_y = 1/args.epoch
        args.tol_v = 1e-10

    args.v = None
    
    # Define paths
    train_path = 'data/spdface_400_inter_histeq/train'
    val_path = 'data/spdface_400_inter_histeq/val'

    # Initialize lists to store data and labels
    data_X = []
    y = []
    data_X_test = []
    test_y = []

    # Load training and validation data
    print("Loading training data...")
    load_data_from_directory(train_path, data_X, y)
    print("Loading validation data...")
    load_data_from_directory(val_path, data_X_test, test_y)

    # Convert lists to numpy arrays
    data_X = np.array(data_X)
    y = np.array(y)
    data_X_test = np.array(data_X_test)
    test_y = np.array(test_y)

    data_X = torch.from_numpy(data_X).float()
    data_y = torch.from_numpy(y-1).long()
    data_X_test = torch.from_numpy(data_X_test).float()
    data_y_test = torch.from_numpy(test_y-1).long()

    # Print dataset statistics before filtering
    print(f"Original dataset size - Training: {len(data_X)}, Testing: {len(data_X_test)}")
    print(f"Original unique labels - Training: {torch.unique(data_y)}, Testing: {torch.unique(data_y_test)}")

    # Define which labels to keep
    i1, i2, i3, i4, i5, i6, i7 = 0, 1, 2, 3, 4, 5, 6,  # Labels to keep
    labels_to_keep = [i1, i2, i3, i4, i5, i6, i7]

    # Filter training data
    train_mask = torch.zeros(data_y.shape, dtype=torch.bool)
    for label in labels_to_keep:
        train_mask = train_mask | (data_y == label)
        
    data_X = data_X[train_mask]
    data_y = data_y[train_mask]

    # Filter test data
    test_mask = torch.zeros(data_y_test.shape, dtype=torch.bool)
    for label in labels_to_keep:
        test_mask = test_mask | (data_y_test == label)
        
    data_X_test = data_X_test[test_mask]
    data_y_test = data_y_test[test_mask]

    # Create mapping from original labels to new consecutive labels
    label_mapping = {original: idx for idx, original in enumerate(labels_to_keep)}

    # Map labels to 0, 1
    data_y = torch.tensor([label_mapping[label.item()] for label in data_y])
    data_y_test = torch.tensor([label_mapping[label.item()] for label in data_y_test])

    # Print dataset statistics after filtering and mapping
    print(f"Filtered dataset size - Training: {len(data_X)}, Testing: {len(data_X_test)}")
    print(f"Filtered unique labels - Training: {torch.unique(data_y)}, Testing: {torch.unique(data_y_test)}")
    print(f"Label mapping: {label_mapping}")

    num_class = len(labels_to_keep)
    lam = 0.1

    # Generate random seeds for multiple runs
    np.random.seed(args.seed)
    seeds = np.random.randint(0, 10000, size=args.runs).tolist()
    print(f"Using seeds: {seeds}")

    for data_ratio in [0.125, 0.25]: # , 0.5
        print(f"Training with {data_ratio} of the data")
        acc_all_runs = []
        test_acc_all_runs = []
        loss_all_runs = []
        runtime_all_runs = []
        val50_time_runs = []
        val70_time_runs = []
        val85_time_runs = []

        for run in range(args.runs):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(seeds[run])
            print(device)
            torch.random.manual_seed(seeds[run])
            np.random.seed(seeds[run])
            torch.backends.cudnn.deterministic = True

            train_X = []
            val_X = []
            train_y = []
            val_y = []

            for c in labels_to_keep:
                data_sub = data_X[data_y == c]
                idx_all =  torch.randperm(data_sub.shape[0])
                train_idx = idx_all[:int(idx_all.shape[0] * data_ratio)]
                val_idx = idx_all[int(idx_all.shape[0] * data_ratio):int(idx_all.shape[0]*2*data_ratio)]
                train_X.append(data_sub[train_idx])
                val_X.append(data_sub[val_idx])

                train_y.extend([c]*train_idx.shape[0])
                val_y.extend([c]*val_idx.shape[0])

            train_X = torch.cat(train_X, dim=0)
            val_X = torch.cat(val_X, dim=0)
            train_y = torch.tensor(train_y, dtype=torch.long)
            val_y = torch.tensor(val_y, dtype=torch.long)

            idx_train = torch.randperm(train_y.shape[0])
            train_X = (train_X[idx_train]).to(device)
            train_y = train_y[idx_train].to(device)
            idx_val = torch.randperm(val_y.shape[0])
            val_X = val_X[idx_val].to(device)
            val_y = val_y[idx_val].to(device)
            test_X = data_X_test.to(device)
            test_y = data_y_test.to(device)


            euclidean = EuclideanMod(ndim=2)
            stiefel = Stiefel(canonical=False)

            d = 400
            d1 = 100
            d2 = 20
            d3 = 5

            print('initialization with manifold network')
            hparams = [ManifoldParameter(stiefel.random(d, d1, device=device), manifold=stiefel),
                ManifoldParameter(stiefel.random(d1, d2, device=device), manifold=stiefel),
                ManifoldParameter(stiefel.random(d2, d3, device=device), manifold=stiefel)]
            params = [ManifoldParameter(euclidean.random(int(d3 * (d3 + 1) / 2), num_class, device=device), manifold=euclidean)]
            mfd_params = [param.manifold for param in params]
            loss_upper = loss_upper_mfd
            loss_lower = loss_lower_mfd
            network = SPDnet

            data_lower = [train_X, train_y]
            data_upper = [val_X, val_y]
            data_test = [test_X, test_y]

            epochs_all = [0]
            loss_u_all = [loss_upper(hparams, params, data_upper).item()]
            runtime = [0]
            acc_all = [compute_acc(network, hparams, params, data_upper).item()]
            test_acc_all = [compute_acc(network, hparams, params, data_test).item()]
            print(f"trainning starts")

            start_time = time.time()
            val50_time = None
            val70_time = None
            val85_time = None
            total_hgradnorm0 = None
            args.alpha = alpha0

            for ep in range(1, args.epoch+1):
                try:
                    hparams, params, args.v, loss_u, hgradnorm, step_time, hg_error, inner_iter, hygrad_flag = AdaRHDstep(loss_lower, loss_upper, hparams, params, args,
                                                                            data=[data_lower, data_upper]) 

                    total_hgradnorm = hgradnorm * len(hparams)
                    if ep == 1:
                        total_hgradnorm0 = total_hgradnorm
                    elif total_hgradnorm > 5 * total_hgradnorm0:
                        hygrad_flag = False

                    if hygrad_flag:
                        args.alpha = np.sqrt(args.alpha ** 2 + total_hgradnorm)

                    with torch.no_grad():
                        val_acc = compute_acc(network, hparams, params, data_upper)
                        test_acc = compute_acc(network, hparams, params, data_test)
                    
                    loss_u_all.append(loss_u)
                    runtime.append(step_time)
                    epochs_all.append(ep)
                    acc_all.append(val_acc.item())
                    test_acc_all.append(test_acc.item())

                    print(f"Epoch {ep}: "
                            f"loss upper: {loss_u:.4e}, "
                            f"hgradnorm: {hgradnorm:.4e}, "
                            f"alpha: {args.alpha:.4e}, "
                            f"Val acc: {val_acc*100:.2f}, "
                            f"Test acc: {test_acc*100:.2f}, "
                            f"runtime: {step_time:.2f}")
                    
                    # Record when validation accuracy reaches specific thresholds
                    current_time = time.time() - start_time
                    if val50_time is None and val_acc >= 0.50:
                        val50_time = current_time
                        print(f"Validation accuracy reached 50% at {val50_time:.2f} seconds")
                    if val70_time is None and val_acc >= 0.70:
                        val70_time = current_time
                        print(f"Validation accuracy reached 70% at {val70_time:.2f} seconds")
                    if val85_time is None and val_acc >= 0.85:
                        val85_time = current_time
                        print(f"Validation accuracy reached 85% at {val85_time:.2f} seconds")

                    if total_hgradnorm < 1e-3:
                        print(f'AdaRHD converged at epoch {ep} with hypergrad norm {total_hgradnorm}. ')
                        break

                    if ep % 5 == 0:
                        args.y_subiter = min(args.y_subiter + 50, 500)
                        if args.hygrad_opt == 'gd':
                            args.v_subiter = min(args.v_subiter + 50, 500)
                except Exception as e:
                    print(f"Error during AdaRHD step: {e}")
                    break 

            acc_all_runs.append(torch.tensor(acc_all).unsqueeze(0))
            test_acc_all_runs.append(torch.tensor(test_acc_all).unsqueeze(0))
            loss_all_runs.append(torch.tensor(loss_u_all).unsqueeze(0))
            runtime_all_runs.append(torch.tensor(runtime).unsqueeze(0))
            if val50_time is not None:
                val50_time_runs.append(torch.tensor(val50_time).unsqueeze(0))
            else:
                val50_time_runs.append(None)
            if val70_time is not None:
                val70_time_runs.append(torch.tensor(val70_time).unsqueeze(0))
            else:
                val70_time_runs.append(None)
            if val85_time is not None:
                val85_time_runs.append(torch.tensor(val85_time).unsqueeze(0))
            else:
                val85_time_runs.append(None)

        stats = {'runtime': runtime_all_runs, 'loss_upper': loss_all_runs,
                    'accuracy': acc_all_runs, 'test_accuracy': test_acc_all_runs,
                    'val50_time': val50_time_runs, 'val70_time': val70_time_runs, 'val85_time': val85_time_runs} 
        
        res_folder = './results/'
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        
        filename = res_folder + 'hyrep_spd_AdaRHD_' + str(args.hygrad_opt) + '_lr' + str(1 / args.beta)  + 'data_ratio' + str(data_ratio) + '.pkl'

        with open(filename, 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)