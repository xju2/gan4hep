"""
GNN-based discriminator and generator
"""
from typing import Callable, Iterable, Optional, Text

import tensorflow as tf
import sonnet as snt

import tensorflow as tf
from graph_nets import utils_tf
import sonnet as snt

from gan4hep.gan_base import GANBase
from gan4hep.reader import n_max_nodes
from gan4hep.gnn_model import Classifier


class GNN(snt.Module):
    def __init__(self, latent_size=512, num_layers=5):
        super().__init__(name="Generator")
        self._gnn = Classifier()

    def __call__(self, input_op, training: bool = True) -> tf.Tensor:
        return self._gnn(input_op)


class GAN(GANBase):
    def __init__(self, noise_dim, batch_size, latent_size=512, num_layers=10, name=None):
        super().__init__(noise_dim, batch_size, name=name)
        self.generator = GNN(latent_size=latent_size, num_layers=num_layers)
        self.discriminator = GNN(latent_size=latent_size, num_layers=num_layers)

    def generate(self, cond_inputs, is_training=True):
        inputs = self.create_ganenerator_inputs(cond_inputs)
        output = self.generator(inputs, is_training)

        # utils_tf.stop_gradient(output)
        n_nodes = output.n_nodes
        first_node_pos = tf.convert_to_tensor(np.array([i for x in n_nodes for i in [1]+[0]*(x-1)]))
        first_node_neg = tf.convert_to_tensor(np.array([i for x in n_nodes for i in [0]+[1]*(x-1)]))

        nodes = output.nodes * first_node_neg + tf.repeat(cond_inputs, repeats=n_nodes) * first_node_pos
        return nodes
        

    def discriminate(self, inputs, is_training=True):
        return self.discriminator(inputs, is_training)