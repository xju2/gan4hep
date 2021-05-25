import tensorflow as tf
import numpy as np

from gan4hep.graph import loop_dataset
from gan4hep.graph import read_dataset

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