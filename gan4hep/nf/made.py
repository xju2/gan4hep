"""Masked Autoregressive Density Estimation
"""
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras

class Made(tfk.layers.Layer):
    def __init__(self, params,
        event_shape=None,
        conditional=False,
        conditional_event_shape=None,
        hidden_units=None, activation=None,
        use_bias=True, kernel_regularizer=None,
        bias_regularizer=None, name="made") -> None:
        super().__init__(name=name)

        self.params = params
        self.event_shape = event_shape
        self.hidden_units = hidden_units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.network = tfb.AutoregressiveNetwork(
            params=params,
            event_shape=event_shape,
            conditional=conditional,
            conditional_event_shape=conditional_event_shape,
            hidden_units=hidden_units,
            activation=activation,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer, 
            bias_regularizer=bias_regularizer)

    def call(self, x):
        shift, log_scale = tf.unstack(self.network(x), num=2, axis=-1)

        return shift, tf.math.tanh(log_scale)


def create_flow(hidden_shape: list, layers: int, input_dim: int, out_dim: int=2):
    """Create Masked Autogressive Flow for density estimation

    Arguments:
    hidden_shape -- Multilayer Perceptron shape
    layers -- Number of bijectors
    out_dim -- output dimensions
    """

    base_dist = tfd.Normal(loc=0.0, scale=1.0)
    bijectors = []
    permutation = tf.cast(np.concatenate((
        np.arange(input_dim / 2, input_dim), np.arange(0, input_dim / 2))), tf.int32)
    for _ in range(layers):
        bijectors.append(tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=Made(
                out_dim, hidden_units=hidden_shape, activation='relu')
        ))
        bijectors.append(tfb.Permute(permutation=permutation))

    bijector = tfb.Chain(bijectors=list(reversed(bijectors)), name='chain_of_MAF')

    maf = tfd.TransformedDistribution(
        distribution=tfd.Sample(base_dist, sample_shape=[input_dim]),
        bijector=bijector,
    )

    return maf


def create_conditional_flow(
        hidden_shape: list, layers: int,
        conditional_event_shape: tuple,
        out_dim=2):
    """Create Conditional Masked Autogressive Flow for density estimation

    Arguments:
    hidden_shape -- Multilayer Perceptron shape
    layers -- Number of bijectors
    conditional_event_shape -- dimentionality of conditions
    out_dim -- output dimensions
    """

    base_dist = tfd.Normal(loc=0.0, scale=1.0)
    bijectors = []

    for i in range(layers):
        bijectors.append(tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=Made(
                out_dim,
                conditional=True,
                conditional_event_shape=conditional_event_shape,
                hidden_units=hidden_shape, activation='relu'),
            name=f"b{i}"
        ))
        bijectors.append(tfb.Permute(permutation=[1, 0]))

    bijector = tfb.Chain(
        bijectors=list(reversed(bijectors)), name='chain_of_MAF')

    maf = tfd.TransformedDistribution(
        distribution=tfd.Sample(base_dist, sample_shape=[out_dim]),
        bijector=bijector,
    )

    return maf