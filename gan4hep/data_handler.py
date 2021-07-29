import tensorflow as tf
import numpy as np
import sonnet as snt
from gan4hep.graph import read_dataset


n_node_features = 6
n_max_nodes = 2 # 

max_energy_px_py_pz = np.array([49.1, 47.7, 46.0, 47.0], dtype=np.float32)
max_energy_px_py_pz_HI = np.array([10]*4, dtype=np.float32)
max_pt_eta_phi_energy = np.array([5, 5, np.pi, 5], dtype=np.float32)

node_mean = np.array([
    [14.13, 0.05, -0.10, -0.04], 
    [7.73, 0.02, -0.04, -0.08],
    [6.41, 0.04, -0.06, 0.04]
], dtype=np.float32)

node_scales = np.array([
    [13.29, 10.54, 10.57, 12.20], 
    [8.62, 6.29, 6.35, 7.29],
    [6.87, 5.12, 5.13, 5.90]
], dtype=np.float32)


node_abs_max = np.array([
    [49.1, 47.7, 46.0, 47.0],
    [46.2, 40.5, 41.0, 39.5],
    [42.8, 36.4, 37.0, 35.5]
], dtype=np.float32)


def get_pt_eta_phi(px, py, pz):
    p = np.sqrt(px**2 + py**2 + pz**2)
    pt = np.sqrt(px**2 + py**2)
    phi = np.arctan2(py, px)
    theta = np.arccos(pz/p)
    eta = -np.log(np.tan(0.5*theta))
    return pt,eta,phi


def normalize(inputs, targets, batch_size, to_tf_tensor=True, hadronic=False, use_pt_eta_phi_e=False):
    scales = max_energy_px_py_pz_HI if hadronic else max_energy_px_py_pz

    # node features are [energy, px, py, pz]
    if use_pt_eta_phi_e:
        # inputs
        pt, eta, phi = get_pt_eta_phi(inputs.nodes[:, 1], inputs.nodes[:, 2], inputs.nodes[:, 3])
        input_nodes = np.stack([pt, eta, phi, inputs.nodes[:, 0]], axis=1) / max_pt_eta_phi_energy
        # outputs
        o_pt, o_eta, o_phi = get_pt_eta_phi(targets.nodes[:, 1], targets.nodes[:, 2], targets.nodes[:, 3])
        target_nodes = np.stack([o_pt, o_eta, o_phi, targets.nodes[:, 0]], axis=1) / max_pt_eta_phi_energy
    else:
        # input_nodes = (inputs.nodes - node_mean[0])/node_scales[0]
        input_nodes = inputs.nodes / scales
        target_nodes = targets.nodes / scales

    target_nodes = np.reshape(target_nodes, [batch_size, -1])
    if hadronic:
        target_nodes = target_nodes[..., :4*3]*np.array([1e4]*3+[0.1]+[1.0]*8)
        input_nodes = input_nodes*np.array([1e4, 1e4, 1e4, 0.1])

    if to_tf_tensor:
        input_nodes = tf.convert_to_tensor(input_nodes, dtype=tf.float32)
        target_nodes = tf.convert_to_tensor(target_nodes, dtype=tf.float32)
    return input_nodes, target_nodes


def read_tfdata(filename, evts_per_record):
    dataset, n_graphs = read_dataset(filename, evts_per_record)
    truth_4vec = []
    input_4vec = []
    for inputs, targets in dataset:
        input_4vec.append(inputs.nodes)
        truth_4vec.append(tf.reshape(targets.nodes, [1, -1, 4]).numpy())
        
    truth_4vec = np.concatenate(truth_4vec, axis=0)
    input_4vec = np.concatenate(input_4vec)
    return input_4vec, truth_4vec


def xyz2hep(px, py, pz):
    p = np.sqrt(px**2 + py**2 + pz**2)
    pt = np.sqrt(px**2 + py**2)
    phi = np.arctan2(py, px)
    theta = np.arccos(pz/p)
    eta = -np.log(np.tan(0.5*theta))
    return pt,eta,phi