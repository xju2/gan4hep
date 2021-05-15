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

NUM_LAYERS = 2
LATENT_SIZE = 256
def make_mlp_model():
  """Instantiates a new MLP, followed by LayerNorm.
  The parameters of each new MLP are not shared with others generated by
  this function.
  Returns:
    A Sonnet module which contains the MLP and LayerNorm.
  """
  return snt.Sequential([
      snt.nets.MLP([LATENT_SIZE]*NUM_LAYERS,
                   activation=tf.nn.leaky_relu,
                   activate_final=True),
      snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
  ])

class MLPGraphNetwork(snt.Module):
    """GraphIndependent with MLP edge, node, and global models."""
    def __init__(self, name="MLPGraphNetwork"):
        super(MLPGraphNetwork, self).__init__(name=name)
        self._network = modules.GraphNetwork(
            edge_model_fn=make_mlp_model,
            node_model_fn=make_mlp_model,
            global_model_fn=make_mlp_model
            )

    def __call__(self, inputs,
            edge_model_kwargs=None,
            node_model_kwargs=None,
            global_model_kwargs=None):
        return self._network(inputs,
                      edge_model_kwargs=edge_model_kwargs,
                      node_model_kwargs=node_model_kwargs,
                      global_model_kwargs=global_model_kwargs)

class NodeBasedGraphEncoder(snt.Module):
  def __init__(self, name="NodeBasedGraphEncoder"):
    super().__init__(name=name)

    self._node_encoder_block = blocks.NodeBlock(
        node_model_fn=make_mlp_model,
        use_received_edges=False,
        use_sent_edges=False,
        use_nodes=True,
        use_globals=False,
        name='node_encoder_block'
    )
    self._edge_block = blocks.EdgeBlock(
        edge_model_fn=make_mlp_model,
        use_edges=False,
        use_receiver_nodes=True,
        use_sender_nodes=True,
        use_globals=False,
        name='edge_encoder_block'
    )
    self._global_encoder_block = blocks.GlobalBlock(
        global_model_fn=make_mlp_model, use_globals=False)

  def __call__(self, inputs):
    return self._global_encoder_block(
      self._edge_block(self._node_encoder_block(inputs)))


class GlobalBasedGraphEncoder(snt.Module):
  def __init__(self, name="GlobalBasedGraphEncoder"):
    super().__init__(name=name)

    self._global_encoder_block = blocks.GlobalBlock(
        global_model_fn=make_mlp_model, use_globals=True,
        use_nodes=False, use_edges=False,
        name='global_encoder_block'
    )

    self._node_encoder_block = blocks.NodeBlock(
        node_model_fn=make_mlp_model,
        use_received_edges=False,
        use_sent_edges=False,
        use_nodes=True,
        use_globals=True,
        name='node_encoder_block'
    )

    self._edge_block = blocks.EdgeBlock(
        edge_model_fn=make_mlp_model,
        use_edges=False,
        use_receiver_nodes=True,
        use_sender_nodes=True,
        use_globals=True,
        name='edge_encoder_block'
    )

  def __call__(self, inputs):
    return self._edge_block(
      self._node_encoder_block(self._global_encoder_block(inputs)))


class Classifier(snt.Module):

  def __init__(self, name="Classifier"):
    super().__init__(name=name)

    self._encoder = NodeBasedGraphEncoder()
    self._core = MLPGraphNetwork()

    # Transforms the outputs into appropriate shapes.
    node_output_size = 4
    node_fn = lambda: snt.nets.MLP([node_output_size],
                    activate_final=False,
                    name='edge_output')
    global_output_size = 1
    glob_fn = lambda: snt.nets.MLP([global_output_size],
                    activate_final=False,
                    name='global_output')

    self._output_transform = modules.GraphIndependent(None, node_fn, glob_fn)

  def __call__(self, input_op, num_processing_steps=4):
    latent = self._encoder(input_op)

    latent0 = latent
    for _ in range(num_processing_steps):
        core_input = utils_tf.concat([latent0, latent], axis=1)
        latent = self._core(core_input)

    return self._output_transform(latent)


class GlobalBasedClassifier(snt.Module):

  def __init__(self, name="GlobalBasedClassifier"):
    super().__init__(name=name)

    self._encoder = GlobalBasedGraphEncoder()

    self._core = MLPGraphNetwork()

    # Transforms the outputs into appropriate shapes.
    node_output_size = 4
    node_fn = lambda: snt.nets.MLP([node_output_size],
                    activate_final=False,
                    name='edge_output')
    global_output_size = 1
    glob_fn = lambda: snt.nets.MLP([global_output_size],
                    activate_final=False,
                    name='global_output')

    self._output_transform = modules.GraphIndependent(None, node_fn, glob_fn)

  def __call__(self, input_op, num_processing_steps=4):
    latent = self._encoder(input_op)

    latent0 = latent
    for _ in range(num_processing_steps):
        core_input = utils_tf.concat([latent0, latent], axis=1)
        latent = self._core(core_input)
    
    return self._output_transform(latent)
