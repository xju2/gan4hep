import numpy as np

from gan4hep.io import GAN_INPUT_DATA_TYPE
from gan4hep.io.utils import shuffle


def read_cluster_decay(filename,
        max_evts=None,
        testing_frac=0.1) -> GAN_INPUT_DATA_TYPE:
    """
    This Herwig dataset is for the "ClusterDecayer" study.
    Each event has q1, q1, cluster, h1, h2.
    I define 3 modes:
    0) both q1, q2 are with Pert=1
    1) only one of q1 and q2 is with Pert=1
    2) neither q1 nor q2 are with Pert=1
    3) at least one quark with Pert=1
    """
    if type(filename) == list:
        print(len(filename),"too many files")
        filename = filename[0]
    arrays = np.load(filename)
    truth_in = arrays['out_truth']
    input_4vec = arrays['input_4vec']

    shuffle(truth_in)
    shuffle(input_4vec)
    print(truth_in.shape, input_4vec.shape)

    # Split the data into training and testing
    # <HACK, FIXME, NOTE>
    # <HACK, For now a maximum of 10,000 events are used for testing, xju>
    num_test_evts = int(input_4vec.shape[0]*testing_frac)
    if num_test_evts < 10_000:
        print("WARNING: num_test_evts < 10_000")

    test_in, train_in = input_4vec[:num_test_evts], input_4vec[num_test_evts:max_evts]
    test_truth, train_truth = truth_in[:num_test_evts], truth_in[num_test_evts:max_evts]
    xlabels = ['phi', 'theta']

    return (train_in, train_truth, test_in, test_truth, xlabels)
