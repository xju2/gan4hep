from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""
* Interaction Network:
* node_block: use_received_edges=False
* edge_encoder:  MLP = [latent_size]*2
* edge_output: only [1]
"""

import tensorflow as tf
from graph_nets import modules
from graph_nets import utils_tf
from graph_nets import blocks
from graph_nets.graphs import GraphsTuple 
import sonnet as snt
from functools import partial


def make_mlp_model(
    num_layers: int = 10,
    latent_size: int = 512,
    dropout_rate: float = 0.05,
    activations=tf.nn.leaky_relu,
    activate_final: bool =True,
    create_scale: bool = True,
    create_offset: bool = True,
    name: str = 'MLP', *args, **kwargs):
    return snt.Sequential([
        snt.nets.MLP([latent_size]*num_layers,
                activation=activations,
                activate_final=activate_final, 
                dropout_rate=dropout_rate),
        snt.LayerNorm(axis=-1, create_scale=create_scale, create_offset=create_offset)
        ], name=name)


class MLPGraphNetwork(snt.Module):
    """GraphIndependent with MLP edge, node, and global models."""
    def __init__(self, latent_size=512, num_layers=10, name="MLPGraphNetwork", **kwargs):
        super(MLPGraphNetwork, self).__init__(name=name)
        mlp_fn = partial(make_mlp_model, num_layers=num_layers, latent_size=latent_size, **kwargs)
        self._network = modules.GraphNetwork(
            edge_model_fn=mlp_fn,
            node_model_fn=mlp_fn,
            global_model_fn=mlp_fn
            )

    def __call__(self, inputs, is_training=True):
        edge_kwargs = node_kwargs = global_kwargs = dict(is_training=is_training)
        return self._network(inputs,
                      edge_model_kwargs=edge_kwargs,
                      node_model_kwargs=node_kwargs,
                      global_model_kwargs=global_kwargs)

class NodeBasedGraphEncoder(snt.Module):
    def __init__(self, latent_size=512, num_layers=10, name="NodeBasedGraphEncoder"):
        super().__init__(name=name)
        mlp_fn = partial(make_mlp_model, num_layers=num_layers, latent_size=latent_size)
        self._node_encoder_block = blocks.NodeBlock(
            node_model_fn=mlp_fn,
            use_received_edges=False,
            use_sent_edges=False,
            use_nodes=True,
            use_globals=False,
            name='node_encoder_block'
        )
        self._edge_block = blocks.EdgeBlock(
            edge_model_fn=mlp_fn,
            use_edges=False,
            use_receiver_nodes=True,
            use_sender_nodes=True,
            use_globals=False,
            name='edge_encoder_block'
        )
        self._global_encoder_block = blocks.GlobalBlock(
            global_model_fn=mlp_fn, use_globals=False)

    def __call__(self, inputs, is_training=True):
        edge_kwargs = node_kwargs = global_kwargs = dict(is_training=is_training)
        return self._global_encoder_block(
                    self._edge_block(self._node_encoder_block(inputs, node_kwargs), edge_kwargs),
               global_kwargs)


class GlobalBasedGraphEncoder(snt.Module):
    def __init__(self, latent_size=512, num_layers=10, name="GlobalBasedGraphEncoder"):
        super().__init__(name=name)

        mlp_fn = partial(make_mlp_model, num_layers=num_layers, latent_size=latent_size)

        self._global_encoder_block = blocks.GlobalBlock(
            global_model_fn=mlp_fn, use_globals=True,
            use_nodes=False, use_edges=False,
            name='global_encoder_block'
        )

        self._node_encoder_block = blocks.NodeBlock(
            node_model_fn=mlp_fn,
            use_received_edges=False,
            use_sent_edges=False,
            use_nodes=True,
            use_globals=True,
            name='node_encoder_block'
        )

        self._edge_block = blocks.EdgeBlock(
            edge_model_fn=mlp_fn,
            use_edges=False,
            use_receiver_nodes=True,
            use_sender_nodes=True,
            use_globals=True,
            name='edge_encoder_block'
        )

    def __call__(self, inputs, is_training=True):
        edge_kwargs = node_kwargs = global_kwargs = dict(is_training=is_training)
        return self._edge_block(
            self._node_encoder_block(
                self._global_encoder_block(inputs, global_kwargs), node_kwargs), edge_kwargs)

class Classifier(snt.Module):

    def __init__(self, latent_size=512, num_layers=10, node_output_size=4, name="Classifier"):
        super().__init__(name=name)

        self._encoder = NodeBasedGraphEncoder(latent_size, num_layers)
        self._core = MLPGraphNetwork(latent_size, num_layers)

        # Transforms the outputs into appropriate shapes.
        node_fn = lambda: snt.nets.MLP([latent_size//2, node_output_size],
                        activate_final=False,
                        name='edge_output')
        global_output_size = 1
        glob_fn = lambda: snt.nets.MLP([latent_size//2, global_output_size],
                        activate_final=False,
                        name='global_output')

        self._output_transform = modules.GraphIndependent(None, node_fn, glob_fn)

    def __call__(self, input_op, num_processing_steps=4, is_training=True):
        latent = self._encoder(input_op, is_training)

        latent0 = latent
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input, is_training)

        return self._output_transform(latent)