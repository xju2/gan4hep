"""
RNN-based Generator and MLP-based Discriminator
"""
from typing import Callable, Iterable, Optional, Text

import tensorflow as tf
import sonnet as snt

from gan4hep.gan_base import GANBase
# from gan4hep.reader import n_max_nodes


class CommonRNN(snt.Module):
    def __init__(self,
        is_gen, name,
        max_nodes=2, latent_size=512, num_layers=2,
        activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.leaky_relu
    ):
        """
        initilize the generator by giving the input dimension and output dimension.
        """
        super().__init__(name=name)

        self._node_linear = snt.nets.MLP(
            [latent_size]*num_layers, activation=activation,
            activate_final=True, dropout_rate=None,
            name="node_encoder")

        # deep RNN
        self._node_rnn = snt.DeepRNN([
            snt.LSTM(hidden_size=latent_size),
            snt.LSTM(hidden_size=latent_size)
        ])

        out_dim = 4 if is_gen else 1
        self._node_prop_nn = snt.nets.MLP(
            [latent_size//2]*num_layers+[out_dim], activation=activation,
            activate_final=False, dropout_rate=None,
            name="node_prop_nn")

        self.max_nodes = max_nodes

    def __call__(self,
                 input_op,
                 training: bool = True) -> tf.Tensor:
        """
        Args: 
            input_op: TF.tensor with dimensions [batch-size, n_nodes, features], 
            fetures contain [px, py, pz, E] or 4D noises
            max_nodes: maximum number of output nodes
            training: if in training mode, needed for `dropout`.

        Retruns:
            predicted node featurs with dimension of [batch-size, max-nodes, out-features] 
        """
        input_op = tf.reshape(input_op, [input_op.shape[0], -1, 4])
        batch_size, n_nodes, n_features = input_op.shape
        
        input_op = tf.reshape(input_op, [-1, n_features])
        nodes = self._node_linear(input_op)
        nodes = tf.reshape(nodes, [batch_size, -1, n_features])

        # First node is the incoming particle, use LSTM to generate hidden states
        node_hidden_state = self._node_rnn.initial_state(batch_size)
        _, node_hidden_state = self._node_rnn(
                nodes[:, 0, :], node_hidden_state)

        out_nodes = []
        for inode in range(self.max_nodes):
            # node properties
            node_embedding, node_hidden_state = self._node_rnn(
                nodes[:, inode+1, :], node_hidden_state)
            node_prop = self._node_prop_nn(node_embedding)
            out_nodes.append(node_prop)

        out_nodes = tf.concat(out_nodes, axis=1, name='concat-out-nodes')
        return out_nodes

class Generator(CommonRNN):
        def __init__(self, max_nodes=2, latent_size=512, num_layers=5):
            super().__init__(max_nodes=max_nodes,
                is_gen=True, name="Generator",
                latent_size=latent_size, num_layers=num_layers)

class Discriminator(CommonRNN):
        def __init__(self, max_nodes=2, latent_size=512, num_layers=5):
            super().__init__(max_nodes=max_nodes,
                is_gen=False, name="Discriminator",
                latent_size=latent_size, num_layers=num_layers)

class GAN(GANBase):
    def __init__(self, latent_size=512, num_layers=10, name=None):
        super().__init__(name=name)
        self.generator = Generator(latent_size=latent_size, num_layers=num_layers)
        self.discriminator = Discriminator(latent_size=latent_size, num_layers=num_layers)

    def discriminate(self, inputs, is_training=True):
        return tf.math.reduce_sum(self.discriminator(inputs, is_training), axis=-1, keepdims=True)