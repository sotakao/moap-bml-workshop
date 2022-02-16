import numpy as np
import jax.numpy as jnp
import tensorflow as tf
import matplotlib.pyplot as plt


def plot(X_query, mean, var, samples=None, data=None, title=None, fig=None, figsize=(10,5)):
    """ Function to plot GP predictions
    Args:
        X_query: Tensor of shape [N*, D]
        mean: Tensor of shape [N*, D]
        var: Tensor of shape [N*, D]
        samples: Tensor of shape [S, N*, D]
        data: Tuple (X, Y) where X, Y are tensors of shape [N, D]
        title: String
        fig: matplotlib Figure object
    Note:
        N : number of data points
        N*: number of query (test) points
        D : dimension
        S : number of samples
    """
    if fig == None:
        plt.figure(figsize=figsize)
    plt.plot(X_query, mean, c="C0", lw=2, zorder=2)
    plt.fill_between(
        X_query[:, 0],
        mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
        mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
        color="C0",
        alpha=0.2,
    )
    if samples is not None:
        plt.plot(X_query, samples[:,:,0].numpy().T, "C0", linewidth=0.5, zorder=1)

    if data is not None:
        plt.plot(*data, "xk", zorder=0)

    if title is not None:
        plt.title(title)


def NLLLoss(y, m, v):
    """ Compute the negative log-likelihood loss
    Args:
        y: Tensor of shape [N, D]
        m: Tensor of shape [N, D]
        v: tensor of shape [N, D]
    """
    return tf.math.reduce_mean((y - m)**2/v + tf.math.log(v))


def deg2rad(x: np.ndarray, offset: float = 0.0):
    return (np.pi / 180) * x + offset


def rad2deg(x: np.ndarray, offset: float = 0.0):
    return (180 / np.pi) * (x - offset)


def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)                       # r
    elev = np.arctan2(z,np.sqrt(XsqPlusYsq)) + np.pi/2   # latitude range (-pi/2, pi/2)
    az = np.arctan2(y,x) + np.pi                         # longitude range (-pi, pi)
    return r, elev, az


def set_gp_hyperparameters(key, gp, kernel, lengthscale, amplitude=None):
    (params, state) = gp.init_params_with_state(key)
    sub_kernel_params = params.kernel_params.sub_kernel_params._replace(log_length_scale=jnp.log(lengthscale))
    if amplitude == None: # This sets the amplitude to 1
        pseudodata = jnp.array([0, 0]).reshape(1,2)
        kernel_params = params.kernel_params._replace(sub_kernel_params=sub_kernel_params)
        Kxx = kernel.matrix(kernel_params, pseudodata, pseudodata)
        kernel_params = params.kernel_params._replace(log_amplitude=-jnp.log(Kxx[0, 0, 0, 0]), sub_kernel_params=sub_kernel_params)
    else:
        kernel_params = params.kernel_params._replace(log_amplitude=jnp.log(amplitude), sub_kernel_params=sub_kernel_params)
    params = params._replace(kernel_params=kernel_params)
    return (params, state)
    