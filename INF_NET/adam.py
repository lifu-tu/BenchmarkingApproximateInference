import theano
from collections import OrderedDict
import theano.tensor as T
import lasagne
import numpy as np

def adam(loss, params, learning_rate=0.001, max_norm = 5.0, beta1=0.9,
         beta2=0.999, epsilon=1e-8):
    """Adam updates
    Adam updates implemented
    Parameters
    ----------
    loss : symbolic expressio
        A scalar loss expression
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        Learning rate
    beta1 : float or symbolic scalar
        Exponential decay rate for the first moment estimates.
    beta2 : float or symbolic scalar
        Exponential decay rate for the second moment estimates.
    epsilon : float or symbolic scalar
        Constant for numerical stability.
    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    Notes
    -----
    The paper [1]_ includes an additional hyperparameter lambda. This is only
    needed to prove convergence of the algorithm and has no practical use
    (personal communication with the authors), it is therefore omitted here.
    References
    ----------
    .. [1] Kingma, Diederik, and Jimmy Ba (2014):
           Adam: A Method for Stochastic Optimization.
           arXiv preprint arXiv:1412.6980.
    """
    old_all_grads = theano.grad(loss, params)

    norm = T.sqrt(sum(T.sum(tensor**2) for tensor in old_all_grads))
    target_norm = T.clip(norm, 0, max_norm)
    multiplier = target_norm / (1e-6 + norm)
    
    all_grads = [multiplier*grad0 for grad0 in old_all_grads]	

    t_prev = theano.shared(lasagne.utils.floatX(0.))
    updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    t = t_prev + 1
    a_t = learning_rate*T.sqrt(one-beta2**t)/(one-beta1**t)

    for param, g_t in zip(params, all_grads):
        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_t = beta1*m_prev + (one-beta1)*g_t
        v_t = beta2*v_prev + (one-beta2)*g_t**2
        step = a_t*m_t/(T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t_prev] = t
    return updates
