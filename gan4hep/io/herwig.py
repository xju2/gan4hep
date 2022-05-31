import numpy as np
import pandas as pd

from gan4hep.io import GAN_INPUT_DATA_TYPE
from gan4hep.io.utils import shuffle
from gan4hep.io.utils import read_dataframe
from gan4hep.io.utils import split_to_float

from gan4hep.preprocessing import InputScaler
from gan4hep.preprocessing.booster import boost

def read(filename, max_evts=None, testing_frac=0.1) -> GAN_INPUT_DATA_TYPE:
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
        print(len(filename),"too many files!")
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


def convert_cluster_decay(filename, outname, mode=2, example=False):
    """
    This function reads the original cluster decay files 
    and boost the two hadron decay prodcuts to the cluster frame in which
    they are back-to-back. Save the output to a new file for training the model.
    
    This Herwig dataset is for the "ClusterDecayer" study.
    Each event has q1, q1, cluster, h1, h2.
    I define 3 modes:
    0) both q1, q2 are with Pert=1
    1) only one of q1 and q2 is with Pert=1
    2) neither q1 nor q2 are with Pert=1
    3) at least one quark with Pert=1
    """

    print(f'reading from {filename}')
    df = read_dataframe(filename, ";", 'python')

    q1,q2,c,h1,h2 = [split_to_float(df[idx]) for idx in range(5)]
    if mode not in [0, 1, 2, 3]:
        print(f"mode {mode} is not known! use mode 2")
        mode = 2

    if mode == 0:
        selections = (q1[5] == 1) & (q2[5] == 1)
        print("mode 0: both q1, q2 are with Pert=1")
    elif mode == 1:
        selections = ((q1[5] == 1) & (q2[5] == 0)) | ((q1[5] == 0) & (q2[5] == 1))
        print("mode 1: only one of q1 and q2 is with Pert=1")
    elif mode == 2:
        selections = (q1[5] == 0) & (q2[5] == 0)
        print("mode 2: neither q1 nor q2 are with Pert=1")
    elif mode == 3:
        selections = ~(q1[5] == 0) & (q2[5] == 0)
        print("mode 3: at least one quark with Pert=1")
    else: pass

    outname = outname+f"_mode{mode}"
        
    cluster = c[[1, 2, 3, 4]][selections].values
    h1 = h1[[1, 2, 3, 4]][selections]
    h2 = h2[[1, 2, 3, 4]][selections]

    org_inputs = np.concatenate([cluster, h1, h2], axis=1)

    if example:
        print(org_inputs[0])
        return 

    # print("origin", org_inputs.shape, org_inputs)
    new_inputs = np.array([boost(row) for row in org_inputs])
    # print("boosted", new_inputs.shape, new_inputs)


    out_4vec = new_inputs[:, -4:]
    _,px,py,pz = [out_4vec[:, idx] for idx in range(4)]
    pT = np.sqrt(px**2 + py**2)
    phi = np.arctan(px/py)
    theta = np.arctan(pT/pz)
    out_truth = np.stack([phi, theta], axis=1)

    input_4vec = cluster

    scaler = InputScaler()
    # the input 4vector is the cluster 4vector
    input_4vec = scaler.transform(input_4vec, outname+"_scalar_input4vec.pkl")
    out_truth = scaler.transform(out_truth, outname+"_scalar_outtruth.pkl")

    np.savez(outname, input_4vec=input_4vec, out_truth=out_truth)


def check_converted_data(outname, mode):
    import matplotlib.pyplot as plt
    outname = outname+f"_mode{mode}"

    arrays = np.load(outname+".npz")
    truth_in = arrays['out_truth']
    plt.hist(truth_in[:, 0], bins=100, histtype='step', label='phi')
    plt.hist(truth_in[:, 1], bins=100, histtype='step', label='theta')
    plt.savefig("angles.png")

    scaler_input = InputScaler(outname=outname+"_scalar_input4vec.pkl")
    scaler_output = InputScaler(outname=outname+"_scalar_outtruth.pkl")

    print("//---- inputs ----")
    scaler_input.dump()
    print("//---- output ----")
    scaler_output.dump()
    print("Total entries:", truth_in.shape[0])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='convert herwig decayer')
    add_arg = parser.add_argument
    add_arg('inname', help='input filename')
    add_arg('outname', help='output filename')
    add_arg('-m', '--mode', help='mode', type=int, default=2)
    add_arg("-c", '--check', action='store_true', help="check outputs")
    add_arg("-e", '--example', action='store_true', help='print an example event')
    args = parser.parse_args()
    
    if args.check:
        check_converted_data(args.outname, args.mode)
    else:
        convert_cluster_decay(args.inname, args.outname, args.mode, args.example)