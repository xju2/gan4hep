"""
Generative model based on Graph Neural Network
"""
from types import SimpleNamespace
import functools
from typing import Callable, Iterable, Optional, Text

import tensorflow as tf
import sonnet as snt

from gan4hep.src.models.mpl_gan import Discrimantor
from gan4hep.src.models.mpl_gan import GAN

class Generator(snt.Module):
    def __init__(self, out_dim: int = 4,
        latent_size=512, num_layers=5,
        rnn_latent_size=512,
        activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.leaky_relu,
        with_regulization: bool = True,
        with_batch_norm: bool = False,
        name="Generator",
        ):
        """
        initilize the generator by giving the input dimension and output dimension.
        """
        super().__init__(name=name)

        self._node_linear = snt.nets.MLP(
            [latent_size]*num_layers+[out_dim], activation=activation,
            activate_final=False, dropout_rate=None,
            name="node_encoder")

        # deep RNN
        self._node_rnn = snt.DeepRNN([
            snt.LSTM(hidden_size=rnn_latent_size),
            snt.LSTM(hidden_size=rnn_latent_size)
        ])
        # self._node_rnn = snt.GRU(hidden_size=LATENT_SIZE, name="node_rnn")
        self._node_prop_nn = snt.nets.MLP(
            [latent_size]*num_layers+[out_dim], activation=activation,
            activate_final=False, dropout_rate=None,
            name="node_prop_nn")

        self.out_dim = out_dim

    def __call__(self,
                 input_op,
                 max_nodes: int,
                 training: bool = True) -> tf.Tensor:
        """
        Args: 
            input_op: 2D vector with dimensions [batch-size, features], 
            the latter containing [px, py, pz, E, N-dimension noises]
            max_nodes: maximum number of output nodes
            training: if in training mode, needed for `dropout`.

        Retruns:
            predicted node featurs with dimension of [batch-size, max-nodes, out-features] 
        """

        batch_size = input_op.shape[0]
        node_hidden_state = self._node_rnn.initial_state(batch_size)

        nodes = self._node_linear(input_op)

        nodes = tf.reshape(nodes, [batch_size, 1, self.out_dim])
        for inode in range(max_nodes):

            # node properties
            node_embedding = tf.math.reduce_sum(nodes, axis=1)
            node_embedding, node_hidden_state = self._node_rnn(
                node_embedding, node_hidden_state)
            node_prop = self._node_prop_nn(node_embedding)
            node_prop = tf.reshape(node_prop, [batch_size, 1, self.out_dim])

            # add new node to the existing nodes
            nodes = tf.concat([nodes, node_prop], axis=1, name='add_new_node')
        return tf.reshape(nodes[:, 1:, :], [batch_size, -1])