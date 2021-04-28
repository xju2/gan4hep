"""
GNN-based discriminator and generator
"""
from typing import Callable, Iterable, Optional, Text
import numpy as np

import tensorflow as tf
import sonnet as snt

import tensorflow as tf
from graph_nets import utils_tf
from graph_nets import graphs

import sonnet as snt

from gan4hep.gan_base import GANBase
from gan4hep.reader import n_max_nodes
from gan4hep.gnn_model import Classifier


class GNN(snt.Module):
    def __init__(self, latent_size=512, num_layers=5, name=None):
        super().__init__(name=name)
        self._gnn = Classifier()

    def __call__(self, input_op, training: bool = True) -> tf.Tensor:
        return self._gnn(input_op)


def nodes_to_graph(nodes):
    """
        nodes are of dimensions: [batch_size, 4vectors-of-inputs]
        reshape to [batch_size, n_particles, 4vectors]
        create fully-connected graph from the input nodes
        Now, assuming all events have the same number of particles
    """
    batch_size, n_particles = nodes.shape[0], nodes.shape[1]//4
    
    G = graphs.GraphsTuple(
        nodes=tf.reshape(nodes, [-1, 4]), edges=None, globals=None, receivers=None,
        senders=None,
        n_node=tf.convert_to_tensor(np.array([n_particles]*batch_size), tf.int32),
        n_edge=tf.convert_to_tensor(np.array([0]*batch_size), tf.int32))
    G = utils_tf.fully_connect_graph_static(G)
    return G


def print_graph(G):
    for field_name in graphs.ALL_FIELDS:
        field = getattr(G, field_name)
        if field is not None:
            print(field_name, field.shape)


class GAN(GANBase):
    def __init__(self, noise_dim, batch_size, latent_size=512, num_layers=10, name=None):
        super().__init__(noise_dim, batch_size, name=name)
        self.generator = GNN(latent_size=latent_size, num_layers=num_layers, name='generator')
        self.discriminator = GNN(latent_size=latent_size, num_layers=num_layers, name='discriminator')

    def create_ganenerator_inputs(self, cond_inputs=None):
        inputs = self.get_noise_batch()
        if cond_inputs is not None:
            inputs = tf.concat([cond_inputs, inputs], axis=-1)
        return nodes_to_graph(inputs)

    def generate(self, cond_inputs, is_training=True):
        input_graphs = self.create_ganenerator_inputs(cond_inputs)
        # print_graph(input_graphs)
        # print(input_graphs.n_node)
        # print(input_graphs.n_edge)
        output = self.generator(input_graphs, is_training)
        # print_graph(output)
        

        # utils_tf.stop_gradient(output)
        n_nodes = output.n_node
        tot_nodes = tf.constant([self.batch_size, 1], tf.int32) 
        # print(n_nodes)
        # first_node_pos = tf.convert_to_tensor(np.array([i for x in n_nodes for i in [1]+[0]*(x-1)]))
        first_node_pos = tf.tile(tf.reshape(tf.repeat(np.array([1, 0, 0], np.float32), [4, 4, 4]), [-1, 4]), tot_nodes)
        # print(first_node_pos)
        # first_node_neg = tf.convert_to_tensor(np.array([i for x in n_nodes for i in [0]+[1]*(x-1)]))
        first_node_neg = tf.tile(tf.reshape(tf.repeat(np.array([0, 1, 1], np.float32), [4, 4, 4]), [-1, 4]), tot_nodes)
        # print(first_node_neg)

        # print(output.nodes)
        # print(cond_inputs.shape)
        # print(tf.repeat(cond_inputs,  repeats=n_nodes, axis=0).shape)
        nodes = output.nodes * first_node_neg + tf.repeat(cond_inputs, repeats=n_nodes, axis=0) * first_node_pos
        # print(nodes)

        return tf.reshape(nodes, [self.batch_size, -1])
        

    def discriminate(self, inputs, is_training=True):
        inputs = tf.reshape(inputs, [self.batch_size, -1, 4])
        G = nodes_to_graph(inputs)
        out_graph = self.discriminator(G, is_training)
        return out_graph.globals