# a simple implementation of Riemannian hypergradient descent for bilevel optimization problem
import torch
import time
from .AdaRHD_utils import autograd, compute_hypergrad_v
import geoopt

import torch.optim.optimizer
from geoopt import ManifoldParameter

# structure of the code is derived from https://github.com/andyjm3/rhgd

def AdaRHDstep(loss_lower, loss_upper, hparams, params, args, data=None, true_hessinv=None):
    """
    Adaptive Riemannian Hypergradient Descent (AdaRHD)

    :param loss_lower: Lower-level loss function
    :param loss_upper: Upper-level loss function
    :param hparams: List of hyper-parameters (x), represented as geoopt.ManifoldParameter
    :param params: List of parameters (y), represented as geoopt.ManifoldParameter
    :param args: Arguments including:
        :param alpha: step size for hparams (x)
        :param beta: Initial step size for params (y)
        :param gamma: Initial step size for auxiliary variable (v)
        :param v: Initial auxiliary variable
        :param tol_y: Tolerance for convergence of y subproblem
        :param tol_v: Tolerance for convergence of v subproblem
        :param y_subiter: Number of iterations for y subproblem
        :param v_subiter: Number of iterations for v subproblem
        :param lam: regularization parameter for Conjugate Gradient (CG) method in v subproblem
        :param hygrad_opt: Hypergradient options: {hinv, cg, gd}
        :param compute_hg_error: Flag to compute hypergradient error
        :param verbose: Flag to print verbose logs
    :param data: Optional data input
    :param true_hessinv: True Hessian inverse function
    :return:Updated hparams, params
            v_new: Updated auxiliary variable
            loss_u: Upper-level loss
            hgradnorm: Hypergradient norm
            step_time: Time taken for the step
            hg_error: Hypergradient error
            inner_iter: Number of iterations for y subproblem
            hygrad_flag: Flag to indicate if the hypergradient is computed successfully
    """

    # Initialization
    alpha, beta, gamma = args.alpha, args.beta, args.gamma
    v = args.v
    tol_y, tol_v = args.tol_y, args.tol_v

    assert (isinstance(data, tuple) or isinstance(data, list) or data is None)
    if data is not None:
        if len(data) == 1:
            data_lower = data[0]
            data_upper = data[0]
        elif len(data) == 2:
            data_lower = data[0]
            data_upper = data[1]
    else:
        data_lower = data_upper = None

    def compute_hgradnorm():
        hgradnorm = 0
        for mfd, hparam, hg in zip(mfd_hparams, hparams, hypergrad):
            hgradnorm += mfd.inner(hparam, hg).item() / len(hparams)
        return hgradnorm
    
    def sumls(ls):
        out = 0
        for ll in ls:
            out += ll
        return out
    
    mfd_hparams = [hparam.manifold for hparam in hparams]
    mfd_params = [param.manifold for param in params]

    step_start_time = time.time()

    # Inner loop for parameter updates (y_t)
    inner_iter = 0
    while True and inner_iter < args.y_subiter:
        grad = autograd(loss_lower(hparams, params, data_lower), params)
        with torch.no_grad():
            grad_norm = []
            for param, egrad in zip(params, grad):
                rgrad = param.manifold.egrad2rgrad(param, egrad)
                grad_norm.append(param.manifold.inner(param, rgrad))
                new_param = param.manifold.retr(param, -1/beta * rgrad)
                param.copy_(new_param)
        
        tol_grad_norm = sumls(grad_norm)
        beta = torch.sqrt(beta ** 2 + tol_grad_norm)
        grad_norm_sum = tol_grad_norm / len(params)
        if inner_iter % 10 == 0 and args.verbose:
            print(f"y-subproblem iter: {inner_iter}, grad_norm: {grad_norm_sum}, beta: {beta}")
        
        if grad_norm_sum <= tol_y:
            if args.verbose:
                print("y subproblem converged!")
                print(f"y-subproblem iter: {inner_iter}, grad_norm: {grad_norm_sum}, beta: {beta}")
            break
        
        inner_iter += 1

    # update hypergradient (v_t) (compute hypergrad estimate)
    hypergrad, v_new, hygrad_flag = compute_hypergrad_v(loss_lower, loss_upper, hparams, params, 
                                    data_lower, data_upper, v, hygrad_opt=args.hygrad_opt, true_hessinv=true_hessinv,
                                    iter = args.v_subiter, gd_gamma = gamma, v_reg=args.v_reg, tol=tol_v, verbose=args.verbose)
    
    if args.hygrad_opt == 'cg':
        v_new = None

    # computer error between true hypergradient and estimated hypergradient
    params = [ManifoldParameter(p.detach().clone(), manifold=mfd) for mfd,p in zip(mfd_params,params)]

    with torch.no_grad():
        if args.compute_hg_error and (true_hessinv is not None):
            with torch.enable_grad():
                true_hg, _, _ = compute_hypergrad_v(loss_lower, loss_upper, hparams, params, 
                                    data_lower, data_upper, v, hygrad_opt='hinv', true_hessinv=true_hessinv,
                                    iter = args.v_subiter, gd_gamma = gamma, v_reg=args.v_reg, tol=tol_v, verbose=args.verbose)
            hg_error = [torch.sqrt(hp.manifold.inner(hp, hg-t_hg)).item() for hg, t_hg, hp in zip(hypergrad, true_hg, hparams)]
            hg_error = torch.Tensor(hg_error).sum().item()
        else:
            hg_error = 0
    
    # update hyperparameters (x_t)
    with torch.no_grad():
        for hparam, hg in zip(hparams, hypergrad):
            new_hparam = hparam.manifold.retr(hparam, - 1/alpha * hg)
            hparam.copy_(new_hparam)


        loss_u = loss_upper(hparams, params, data_upper).item()
        hgradnorm = compute_hgradnorm()

    step_time = time.time() - step_start_time

    # deactivate the computational path
    hparams = [geoopt.ManifoldParameter(hparam.detach().clone(), manifold=mfd) for mfd, hparam in zip(mfd_hparams, hparams)]
    params = [geoopt.ManifoldParameter(param.detach().clone(), manifold=mfd) for mfd, param in zip(mfd_params, params)]

    return hparams, params, v_new, loss_u, hgradnorm, step_time, hg_error, inner_iter, hygrad_flag