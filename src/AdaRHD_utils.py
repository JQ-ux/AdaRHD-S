import torch
from geoopt.tensor import ManifoldParameter, ManifoldTensor
import math

# the code of function "dot", "autograd", "batch_egrad2rgrad" and "compute_jvp2" is obtained from https://github.com/andyjm3/rhgd
# deal with list of parameters

def dot(tensors_one, tensors_two):
    """List of tensors in tensors_one, tensors_two"""
    ret = tensors_one[0].new_zeros((1, ), requires_grad=True)
    for t1, t2 in zip(tensors_one, tensors_two):
        ret = ret + torch.sum(t1 * t2)
    return ret

def autograd(outputs, inputs, create_graph=False):
    """Compute gradient of outputs w.r.t. inputs, assuming outputs is a scalar."""
    inputs = tuple(inputs)
    grads = torch.autograd.grad(outputs, inputs, create_graph=create_graph, allow_unused=True)
    return [xx if xx is not None else yy.new_zeros(yy.size()) for xx, yy in zip(grads, inputs)]

def batch_egrad2rgrad(params, egrad):
    return [param.manifold.egrad2rgrad(param, eg) for param, eg in zip(params, egrad)]

@torch.no_grad()
def ts_conjugate_gradient(_hvp, b, base, v0=None, maxiter=200, tol=1e-10, v_reg=0.0, verbose=1):
    """
    Solve H[v] = b where H is a tangent space operator at base points.
    All are list of tensors!

    :param _hvp: H vector product (function that takes a list[Tensor] and outputs a list[Tensor])
    :param b: List[tangent vector]
    :param base: base point (list[geoopt.ManifoldParameter])
    :param v0: initialization (default to be zero)
    :param maxiter: maximum number of iteration
    :param tol: tol for residual norm
    :param lam: regularization strength to ensure positive definiteness
    :param verbose: verbosity level
    :return: solution v to Hreg[v] = b where Hreg = H + v_reg * I
    """

    def hvp(inputs):
        with torch.enable_grad():
            outputs = _hvp(inputs)
        outputs = [xx + v_reg * yy for xx, yy in zip(outputs, inputs)]
        return outputs
    
    def sumls(ls):
        out = 0
        for ll in ls:
            out += ll
        return out

    # Initialize
    if v0 is None:
        v = [hb.new_zeros(hb.size()) for hb in b]
    else:
        v = v0
    r = [hb.clone().detach() for hb in b]
    p = [hb.clone().detach() for hb in b]

    rnormprev = [xx.manifold.inner(xx, rr, rr) for xx, rr in zip(base, r)]

    it = 0
    while True:
        with torch.enable_grad():
            Hp = hvp(p)
        alpha = [rn / xx.manifold.inner(xx, pp, hpp) for rn, xx, pp, hpp in zip(rnormprev, base, p, Hp)]
        v = [vv + aa * pp for vv, aa, pp in zip(v, alpha, p)]
        r = [rr - aa * hpp for rr, aa, hpp in zip(r, alpha, Hp)]
        rnorm = [xx.manifold.inner(xx, rr, rr) for xx, rr in zip(base, r)]
        rnorm_total = sumls(rnorm)
        if rnorm_total < tol:
            if verbose:
                print(f"CG tol reached, break at iter {it}.")
            break
        elif it >= maxiter:
            if verbose:
                print(f"CG tol not reached! Break at max iteration with residual {rnorm_total:.4e}.")
            break
        beta = [rn / rnprev for rn, rnprev in zip(rnorm, rnormprev)]
        p = [rr + bb * pp for rr, bb, pp in zip(r, beta, p)]
        rnormprev = rnorm
        it += 1

    return v

@torch.no_grad()
def ts_gradient_descent(_hvp, b, base, v0=None, maxiter=200, tol=1e-10, gamma=None, v_reg=0.0, verbose=1):
    """
    Solve H[v] = b where H is a tangent space operator at base points,
    Solved via regularized gradient descent: v = v - 1/gamma * (H[v] - b)
    All are list of tensors!

    :param _hvp: H vector product (function that takes a list[Tensor] and output a list[Tensor])
    :param b: List[tangent vector]
    :param v0: Initialization (default to be zero)
    :param base: Base point (list[geoopt.ManifoldParameter])
    :param maxiter: Maximum number of iterations
    :param tol: Tolerance for residual norm
    :param gamma: 1/gamma-Learning rate for gradient descent
    :param lam: Regularization strength to ensure positive definiteness
    :param verbose: Verbosity level (1 for output, 0 for silent)
    :return: Optimized vector v s.t. Hreg[v] = b where Hreg = H + v_reg * I
    """

    def hvp(inputs):
        with torch.enable_grad():
            outputs = _hvp(inputs)
        outputs = [xx + v_reg * yy for xx, yy in zip(outputs, inputs)]  # Regularized operator
        return outputs

    def sumls(ls):
        out = 0
        for ll in ls:
            out += ll
        return out

    # Initialization
    if v0 is None:
        v = b
    else:
        v = v0

    if gamma is None:
        gamma = 1.0

    # Iterative gradient descent
    for it in range(maxiter):
        # Compute H[v]
        with torch.enable_grad():
            Hv = hvp(v)
        # Compute residual: r = H[v] - b
        r = [hv - bb for hv, bb in zip(Hv, b)]
        # Compute norm of residual
        rnorm = [xx.manifold.inner(xx, rr, rr) for xx, rr in zip(base, r)]
        rnorm_total = sumls(rnorm)
        gamma = torch.sqrt(rnorm_total + gamma**2)  # Update learning rate
        # Check for convergence
        if rnorm_total < tol:
            if verbose:
                print(f"Gradient descent tolerance reached, breaking at iteration {it} with residual {rnorm_total:.4e}.")
                print(f"function value: {sumls([0.5*xx.manifold.inner(xx,hv,vv) - xx.manifold.inner(xx,vv,bb) for xx, hv, vv, bb in zip(base,Hv,v,b)])}")
            break
        if it % 10 == 0 and verbose:
            print(f"Gradient descent iteration {it}, residual {rnorm_total:.4e}.")
            print(f"function value: {sumls([0.5*xx.manifold.inner(xx,hv,vv) - xx.manifold.inner(xx,vv,bb) for xx, hv, vv, bb in zip(base,Hv,v,b)])}, gamma: {gamma}")
        # Update v: v = v - 1 /gamma * r
        v = [vv - 1 / gamma * rr for vv, rr in zip(v, r)]
    else:
        if verbose:
            print(f"Gradient descent max iteration reached with residual {sumls(rnorm):.4f}.")
            
    return v

################################################################
def compute_hypergrad_v(loss_lower, loss_upper, hparams, params,
                       data_lower =None, data_upper=None, v0=None, hygrad_opt='cg', true_hessinv=None,
                       iter = 200, gd_gamma = None, v_reg=0.0, tol=1e-8,verbose=1):
    """hparams is x, params is y, loss_lower is g, loss_upper is f

    # :param data_lower: the data for computing the gradient of the lower level problem
    # :param data_upper: the data for computing the gradient of the upper level problem
    # :param gd_iter: the number of iterations for the hypergradient inner optimization
    # :param gd_gamma: the stepsize for the hypergradient inner optimization
    # :param v0: the initial point for the hypergradient inner optimization

    """
    hygrad_flag = True
    # initialize
    egradfy = autograd(loss_upper(hparams, params, data_upper), params)
    egradfx = autograd(loss_upper(hparams, params, data_upper), hparams)
    rgradfy = batch_egrad2rgrad(params, egradfy)
    rgradfx = batch_egrad2rgrad(hparams, egradfx)

    # hessinv grad
    def rhess_prod(u):
        egrad = autograd(loss_lower(hparams, params, data_lower), params, create_graph=True)
        ehess = autograd(dot(egrad, u), params)
        out = []
        with torch.no_grad():
            for idx, param in enumerate(params):
                out.append(param.manifold.ehess2rhess(param, egrad[idx], ehess[idx], u[idx]))
        return out

    if hygrad_opt == 'cg':
        Hinv_gy = ts_conjugate_gradient(rhess_prod, rgradfy, params, v0, v_reg=v_reg, maxiter=iter, tol=tol, verbose=verbose)
    elif hygrad_opt == 'gd':
        Hinv_gy = ts_gradient_descent(rhess_prod, rgradfy, params, v0, gamma=gd_gamma, v_reg=v_reg, maxiter=iter, tol=tol, verbose=verbose)
    elif hygrad_opt == 'hinv':
        assert true_hessinv is not None
        egradfy = autograd(loss_upper(hparams, params, data_upper), params)
        egradfx = autograd(loss_upper(hparams, params, data_upper), hparams)
        rgradfy = batch_egrad2rgrad(params, egradfy)
        rgradfx = batch_egrad2rgrad(hparams, egradfx)

        with torch.no_grad():
            Hinv_gy = true_hessinv(loss_lower, hparams, params, data_lower, rgradfy)

        gradgxy = compute_jvp2(loss_lower, hparams, params, data_lower, Hinv_gy)

        # proj to tangent space (it can be a bit off the tangent space due to numerical errors)
        gradgxy_proj = [hp.manifold.proju(hp, gxy) for hp, gxy in zip(hparams, gradgxy)]

        return [g1 - g2 for g1, g2 in zip(rgradfx, gradgxy_proj)], Hinv_gy, hygrad_flag
    else:
        raise(f"hypergrad option {hygrad_opt} not implemented.")
    
    try:
        gradgxy = compute_jvp2(loss_lower, hparams, params, data_lower, Hinv_gy)
    except:
        # Numerical errors may occur and cause the solving process to fail, so automatic differentiation is adopted to compute the hypergradient.
        print("Warning: JVP failed, use AD instead.")
        egradfy = autograd(loss_upper(hparams, params, data_upper), hparams)
        hygrad_flag = False
        return [hp.manifold.proju(hp, gxy) for hp, gxy in zip(hparams, egradfy)], Hinv_gy, hygrad_flag
    
    # proj to tangent space (it can be a bit off the tangent space due to numerical errors)
    gradgxy_proj = [hp.manifold.proju(hp, gxy) for hp, gxy in zip(hparams,gradgxy)]

    return [g1 - g2 for g1, g2 in zip(rgradfx, gradgxy_proj)], Hinv_gy, hygrad_flag
    
def compute_jvp2(loss, hparams, params, data, tangents):
    """
    Compute the cross derivative of loss(hparams, params), i.e., G_xy [tangents] where x is hparams, y is params
    :param loss:
    :param inputs: List[Tensors] of size hparams
    :param tangents: List[Tensors] of size params
    :return:
    """
    assert len(params) == len(tangents)
    def function(*params):
        grad = autograd(loss(hparams, list(params), data), hparams, create_graph=True) # list of size hparams
        return tuple([hparam.manifold.egrad2rgrad(hparam, gg) for hparam, gg in zip(hparams, grad)])

    _, gradxy = torch.autograd.functional.jvp(function, tuple(params), tuple(tangents))

    return gradxy