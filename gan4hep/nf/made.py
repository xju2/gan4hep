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
        self.use_conditional = conditional
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

    def call(self, x, conditional_input=None):
        if self.use_conditional:
            out = self.network(x, conditional_input=conditional_input)
        else:
            out = self.network(x)
        shift, log_scale = tf.unstack(out, num=2, axis=-1)
        return shift, tf.math.tanh(log_scale)


def create_flow(hidden_shape: list, layers: int,
    input_dim: int,
    with_condition: bool =False,
    conditional_event_shape: tuple = None):
    """Create Masked Autogressive Flow for density estimation

    Arguments:
    hidden_shape -- Multilayer Perceptron shape
    layers -- Number of bijectors
    """
    if with_condition and conditional_event_shape is None:
        raise ValueError("conditional_event_shape must be specified")

    base_dist = tfd.Normal(loc=0.0, scale=1.0)
    permutation = tf.cast(np.concatenate((
        np.arange(input_dim / 2, input_dim), np.arange(0, input_dim / 2))), tf.int32)
    bijectors = []
    for idx in range(layers):
        bijectors.append(tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=Made(
                params=2, event_shape=[input_dim],
                hidden_units=hidden_shape, activation='relu'),
                conditional=with_condition,
                conditional_event_shape=conditional_event_shape,
                name=f"b{idx}"
        ))

        ## https://homepages.inf.ed.ac.uk/imurray2/pub/17maf/maf.pdf
        ## finds that batch normalization reduces training time,
        ## increases stability, and improves performance.
        bijectors.append(tfb.BatchNormalization(training=False))
        if input_dim > 1:
            bijectors.append(tfb.Permute(permutation=permutation))

    # bijectors.append(tfb.Tanh())

    bijector = tfb.Chain(bijectors=list(reversed(bijectors)), name='MAF')

    maf = tfd.TransformedDistribution(
        distribution=tfd.Sample(base_dist, sample_shape=[input_dim]),
        bijector=bijector,
    )

    return maf

