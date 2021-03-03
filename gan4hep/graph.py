"""
Make doublet GraphNtuple
"""
import time
import os
import itertools
import random

import numpy as np
import tensorflow as tf

from graph_nets import utils_tf
from graph_nets import graphs


graph_types = {
    'n_node': tf.int32,
    'n_edge': tf.int32,
    'nodes': tf.float32,
    'edges': tf.float32,
    'receivers': tf.int32,
    'senders': tf.int32,
    'globals': tf.float32,
}

other_features = {
    
}

def padding(g, max_nodes, max_edges, do_concat=True):
    f_dtype = np.float32
    n_nodes = np.sum(g.n_node)
    n_edges = np.sum(g.n_edge)
    n_nodes_pad = max_nodes - n_nodes
    n_edges_pad = max_edges - n_edges

    if n_nodes_pad < 0:
        raise ValueError("Max Nodes: {}, but {} nodes in graph".format(max_nodes, n_nodes))

    if n_edges_pad < 0:
        raise ValueError("Max Edges: {}, but {} edges in graph".format(max_edges, n_edges))

    # padding edges all pointing to the last node
    # TODO: make the graphs more general <xju>
    edges_idx = tf.constant([0] * n_edges_pad, dtype=np.int32)
    # print(edges_idx)
    zeros = np.array([0.0], dtype=f_dtype)
    n_node_features = g.nodes.shape[-1]
    n_edge_features = g.edges.shape[-1]
    # print("input graph global: ", g.globals.shape)
    # print("zeros: ", np.zeros_like(g.globals.numpy()))
    # print("input edges", n_edges, "padding edges:", n_edges_pad)

    padding_datadict = {
        "n_node": n_nodes_pad,
        "n_edge": n_edges_pad,
        "nodes": np.zeros((n_nodes_pad, n_node_features), dtype=f_dtype),
        'edges': np.zeros((n_edges_pad, n_edge_features), dtype=f_dtype),
        'receivers': edges_idx,
        'senders': edges_idx,
        'globals':zeros
    }
    padding_graph = utils_tf.data_dicts_to_graphs_tuple([padding_datadict])
    if do_concat:
        return utils_tf.concat([g, padding_graph], axis=0)
    else:
        return padding_graph

def splitting(g_input, n_devices, verbose=False):
    """
    split the graph so that edges are distributed
    to all devices, of which the number is specified by n_devices.
    """
    def reduce_edges(gg, n_edges_fixed, edge_slice):
        edges = gg.edges[edge_slice]
        return gg.replace(n_edge=tf.convert_to_tensor(np.array([edges.shape[0]]), tf.int32), 
            edges=edges,
            receivers=gg.receivers[edge_slice],
            senders=gg.senders[edge_slice])

    n_edges = tf.math.reduce_sum(g_input.n_edge)
    n_nodes = tf.math.reduce_sum(g_input.n_node)
    splitted_graphs = []
    n_edges_fixed = n_edges // n_devices
    if verbose:
        print("Total {:,} Edges in input graph, splitted into {:,} devices".format(n_edges, n_devices))
        print("Resulting each device contains {:,} edges".format(n_edges_fixed))

    for idevice in range(n_devices):
        if idevice < n_devices - 1:
            edge_slice = slice(idevice*n_edges_fixed, (idevice+1)*n_edges_fixed)
        else:
            edge_slice = slice(idevice*n_edges_fixed, n_edges)
        splitted_graphs.append(reduce_edges(g_input, n_edges_fixed, edge_slice))

    return splitted_graphs


def parse_tfrec_function(example_proto):
    features_description = dict(
        [(key+"_IN",  tf.io.FixedLenFeature([], tf.string)) for key in graphs.ALL_FIELDS] + 
        [(key+"_OUT", tf.io.FixedLenFeature([], tf.string)) for key in graphs.ALL_FIELDS])

    example = tf.io.parse_single_example(example_proto, features_description)
    input_dd = graphs.GraphsTuple(**dict([(key, tf.io.parse_tensor(example[key+"_IN"], graph_types[key]))
        for key in graphs.ALL_FIELDS]))
    out_dd = graphs.GraphsTuple(**dict([(key, tf.io.parse_tensor(example[key+"_OUT"], graph_types[key]))
        for key in graphs.ALL_FIELDS]))
    return input_dd, out_dd

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_graph(G1, G2):
    feature = {}
    for key in graphs.ALL_FIELDS:
        feature[key+"_IN"] = _bytes_feature(tf.io.serialize_tensor(getattr(G1, key)))
        feature[key+"_OUT"] = _bytes_feature(tf.io.serialize_tensor(getattr(G2, key)))
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def concat_batch_dim(G):
    """
    G is a GraphNtuple Tensor, with additional dimension for batch-size.
    Concatenate them along the axis for batch
    It only works for batch size of 1
    """
    n_node = tf.reshape(G.n_node, [-1])
    n_edge = tf.reshape(G.n_edge, [-1])
    nodes = tf.reshape(G.nodes, [-1, tf.shape(G.nodes)[-1]])
    edges = tf.reshape(G.edges, [-1, tf.shape(G.edges)[-1]])
    senders = tf.reshape(G.senders, [-1])
    receivers = tf.reshape(G.receivers, [-1])
    globals_ = tf.reshape(G.globals, [-1, tf.shape(G.globals)[-1]])
    return G.replace(n_node=n_node, n_edge=n_edge, nodes=nodes,\
        edges=edges, senders=senders, receivers=receivers, globals=globals_)


def _concat_batch_dim(G):
    """
    G is a GraphNtuple Tensor, with additional dimension for batch-size.
    Concatenate them along the axis for batch
    """
    input_graphs = []
    for ibatch in [0, 1]:
        data_dict = {
            "nodes": G.nodes[ibatch],
            "edges": G.edges[ibatch],
            "receivers": G.receivers[ibatch],
            'senders': G.senders[ibatch],
            'globals': G.globals[ibatch],
            'n_node': G.n_node[ibatch],
            'n_edge': G.n_edge[ibatch],
        }
        input_graphs.append(graphs.GraphsTuple(**data_dict))
        return (tf.add(ibatch, 1), input_graphs)
    print("{} graphs".format(len(input_graphs)))
    return utils_tf.concat(input_graphs, axis=0)


def add_batch_dim(G, axis=0):
    """
    G is a GraphNtuple Tensor, without a dimension for batch.
    Add a dimensioin them along the axis for batch
    """
    n_node = tf.expand_dims(G.n_node, axis=0)
    n_edge = tf.expand_dims(G.n_edge, axis=0)
    nodes = tf.expand_dims(G.nodes, axis=0)
    edges = tf.expand_dims(G.edges, axis=0)
    senders = tf.expand_dims(G.senders, axis=0)
    receivers = tf.expand_dims(G.receivers, 0)
    globals_ = tf.expand_dims(G.globals, 0)
    return G.replace(n_node=n_node, n_edge=n_edge, nodes=nodes,\
        edges=edges, senders=senders, receivers=receivers, globals=globals_)


def data_dicts_to_graphs_tuple(input_dd, target_dd, with_batch_dim=True):
    # if type(input_dd) is not list:
    #     input_dd = [input_dd]
    # if type(target_dd) is not list:
    #     target_dd = [target_dd]
        
    # input_graphs = utils_tf.data_dicts_to_graphs_tuple(input_dd)
    # target_graphs = utils_tf.data_dicts_to_graphs_tuple(target_dd)
    input_graphs = utils_tf.concat(input_dd, axis=0)
    target_graphs = utils_tf.concat(target_dd, axis=0)
    # # fill zeros
    # input_graphs = utils_tf.set_zero_global_features(input_graphs, 1, dtype=tf.float64)
    # target_graphs = utils_tf.set_zero_global_features(target_graphs, 1, dtype=tf.float64)
    # target_graphs = utils_tf.set_zero_node_features(target_graphs, 1, dtype=tf.float64)
    
    # expand dims
    if with_batch_dim:
        input_graphs = add_batch_dim(input_graphs)
        target_graphs = add_batch_dim(target_graphs)
    return input_graphs, target_graphs


## TODO place holder for the case PerReplicaSpec is exported.
def get_signature(graphs_tuple_sample):
    from graph_nets import graphs

    graphs_tuple_description_fields = {}

    for field_name in graphs.ALL_FIELDS:
        per_replica_sample = getattr(graphs_tuple_sample, field_name)
        def spec_from_value(v):
            shape = list(v.shape)
            dtype = list(v.dtype)
            if shape:
                shape[1] = None
            return tf.TensorSpec(shape=shape, dtype=dtype)

        per_replica_spec = tf.distribute.values.PerReplicaSpec(
            *(spec_from_value(v) for v in per_replica_sample.values)
        )

        graphs_tuple_description_fields[field_name] = per_replica_spec
    return graphs.GraphsTuple(**graphs_tuple_description_fields)


def specs_from_graphs_tuple(
    graphs_tuple_sample, with_batch_dim=False,
    dynamic_num_graphs=False,
    dynamic_num_nodes=True,
    dynamic_num_edges=True,
    description_fn=tf.TensorSpec,
    ):
    graphs_tuple_description_fields = {}
    edge_dim_fields = [graphs.EDGES, graphs.SENDERS, graphs.RECEIVERS]

    for field_name in graphs.ALL_FIELDS:
        field_sample = getattr(graphs_tuple_sample, field_name)
        if field_sample is None:
            raise ValueError(
                "The `GraphsTuple` field `{}` was `None`. All fields of the "
                "`GraphsTuple` must be specified to create valid signatures that"
                "work with `tf.function`. This can be achieved with `input_graph = "
                "utils_tf.set_zero_{{node,edge,global}}_features(input_graph, 0)`"
                "to replace None's by empty features in your graph. Alternatively"
                "`None`s can be replaced by empty lists by doing `input_graph = "
                "input_graph.replace({{nodes,edges,globals}}=[]). To ensure "
                "correct execution of the program, it is recommended to restore "
                "the None's once inside of the `tf.function` by doing "
                "`input_graph = input_graph.replace({{nodes,edges,globals}}=None)"
                "".format(field_name))

        shape = list(field_sample.shape)
        dtype = field_sample.dtype

        # If the field is not None but has no field shape (i.e. it is a constant)
        # then we consider this to be a replaced `None`.
        # If dynamic_num_graphs, then all fields have a None first dimension.
        # If dynamic_num_nodes, then the "nodes" field needs None first dimension.
        # If dynamic_num_edges, then the "edges", "senders" and "receivers" need
        # a None first dimension.
        if shape:
            if with_batch_dim:
                shape[1] = None
            elif (dynamic_num_graphs \
                or (dynamic_num_nodes \
                    and field_name == graphs.NODES) \
                or (dynamic_num_edges \
                    and field_name in edge_dim_fields)): shape[0] = None

        print(field_name, shape, dtype)
        graphs_tuple_description_fields[field_name] = description_fn(
            shape=shape, dtype=dtype)

    return graphs.GraphsTuple(**graphs_tuple_description_fields)


def dtype_shape_from_graphs_tuple(input_graph, with_batch_dim=False, with_padding=True, debug=False, with_fixed_size=False):
    graphs_tuple_dtype = {}
    graphs_tuple_shape = {}

    edge_dim_fields = [graphs.EDGES, graphs.SENDERS, graphs.RECEIVERS]
    for field_name in graphs.ALL_FIELDS:
        field_sample = getattr(input_graph, field_name)
        shape = list(field_sample.shape)
        dtype = field_sample.dtype
        print(field_name, shape, dtype)

        if not with_fixed_size and shape and not with_padding:
            if with_batch_dim:
                shape[1] = None
            else:
                if field_name == graphs.NODES or field_name in edge_dim_fields:
                    shape[0] = None

        graphs_tuple_dtype[field_name] = dtype
        graphs_tuple_shape[field_name] = tf.TensorShape(shape)
        if debug:
            print(field_name, shape, dtype)
    
    return graphs.GraphsTuple(**graphs_tuple_dtype), graphs.GraphsTuple(**graphs_tuple_shape)


def read_dataset(filenames):
    """
    Read dataset...
    """
    AUTO = tf.data.experimental.AUTOTUNE
    tr_filenames = tf.io.gfile.glob(filenames)
    n_files = len(tr_filenames)

    dataset = tf.data.TFRecordDataset(tr_filenames)
    dataset = dataset.map(parse_tfrec_function, num_parallel_calls=AUTO)
    n_graphs = sum([1 for _ in dataset])
    return dataset, n_graphs


def loop_dataset(datasets, batch_size):
    if batch_size > 0:
        in_list = []
        target_list = []
        for dataset in datasets:
            inputs_tr, targets_tr = dataset
            in_list.append(inputs_tr)
            target_list.append(targets_tr)
            if len(in_list) == batch_size:
                inputs_tr = utils_tf.concat(in_list, axis=0)
                targets_tr = utils_tf.concat(target_list, axis=0)
                yield (inputs_tr, targets_tr)
                in_list = []
                target_list = []
    else:
        for dataset in datasets:
            yield dataset