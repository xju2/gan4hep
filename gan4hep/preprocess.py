import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



from pylorentz import Momentum4

def shuffle(array: np.ndarray):
    from numpy.random import MT19937
    from numpy.random import RandomState, SeedSequence
    np_rs = RandomState(MT19937(SeedSequence(123456789)))
    np_rs.shuffle(array)


def read_dataframe(filename, sep=",", engine=None):
    if type(filename) == list:
        print(filename)
        df_list = [
            pd.read_csv(f, sep=sep, header=None, names=None, engine=engine)
                for f in filename
        ]
        df = pd.concat(df_list, ignore_index=True)
        filename = filename[0]
    else:
        df = pd.read_csv(filename, sep=sep, 
                    header=None, names=None, engine=engine)
    return df



   
def herwig_angles(filename,
        max_evts=None, testing_frac=0.1):
    """
    This reads the Herwig dataset where one cluster decays
    into two particles.
    In this case, we ask the GAN to predict the theta and phi
    angle of one of the particles
    """
    df = read_dataframe(filename, engine='python')

    event = None
    with open(filename, 'r') as f:
        for line in f:
            event = line
            break
    particles = event[:-2].split(';')

    input_4vec = df[0].str.split(",", expand=True)[[4, 5, 6, 7]].to_numpy().astype(np.float32)
    out_particles = []
    for idx in range(1, len(particles)):
        out_4vec = df[idx].str.split(",", expand=True).to_numpy()[:, -4:].astype(np.float32)
        out_particles.append(out_4vec)

    # ======================================
    # Calculate the theta and phi angle 
    # of the first outgoing particle
    # ======================================
    out_4vec = out_particles[0]
    px = out_4vec[:, 1].astype(np.float32)
    py = out_4vec[:, 2].astype(np.float32)
    pz = out_4vec[:, 3].astype(np.float32)
    pT = np.sqrt(px**2 + py**2)
    phi = np.arctan(px/py)
    theta = np.arctan(pT/pz)

    # <NOTE, inputs and outputs are scaled to be [-1, 1]>
    max_phi = np.max(np.abs(phi))
    max_theta = np.max(np.abs(theta))
    scales = np.array([max_phi, max_theta], np.float32)

    truth_in = np.stack([phi, theta], axis=1) / scales

    shuffle(truth_in)
    shuffle(input_4vec)


    # Split the data into training and testing
    # <HACK, FIXME, NOTE>
    # <HACK, For now a maximum of 10,000 events are used for testing, xju>
    num_test_evts = int(input_4vec.shape[0]*testing_frac)
    if num_test_evts < 10_000: num_test_evts = 10_000

    # <NOTE, https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html>



    test_in, train_in = input_4vec[:num_test_evts], input_4vec[num_test_evts:max_evts]
    test_truth, train_truth = truth_in[:num_test_evts], truth_in[num_test_evts:max_evts]

    xlabels = ['phi', 'theta']

    return (train_in, train_truth, test_in, test_truth, xlabels)

def herwig_angles2(filename,
        max_evts=None, testing_frac=0.1, mode=2):
    """
    This Herwig dataset is for the "ClusterDecayer" study.
    Each event has q1, q1, cluster, h1, h2.
    I define 3 modes:
    0) both q1, q2 are with Pert=1
    1) either q1 or q2 is with Pert=1
    2) neither q1 nor q2 are with Pert=1
    """
    print(f'reading from {filename}')
    df = read_dataframe(filename, ";", 'python')

    def split_to_float(df, sep=','):
        out = df
        if type(df.iloc[0]) == str:
            out = df.str.split(sep, expand=True).astype(np.float32)
        return out

    q1,q2,c,h1,h2 = [split_to_float(df[idx]) for idx in range(5)]
    if mode not in [0, 1, 2]:
        mode = 2
        print(f"mode {mode} is not known! use mode 2")

    if mode == 0:
        selections = (q1[5] == 1) & (q2[5] == 1)
    elif mode == 1:
        selections = ((q1[5] == 1) & (q2[5] == 0)) | ((q1[5] == 0) & (q2[5] == 1))
    else:
        selections = (q1[5] == 0) & (q2[5] == 0)
        
    # selection hadrons in one of the modes    
    h1 = h1[selections]
    h2 = h2[selections]

    # ======================================
    # Calculate the theta and phi angle, and the energy
    # of the first outgoing particle
    # ======================================
    scaler = MinMaxScaler(feature_range=(-1,1))

    out_4vec = h1.values
    energy,px,py,pz = [out_4vec[:, idx] for idx in range(1,5)]
    pT = np.sqrt(px**2 + py**2)
    phi = np.arctan(px/py)
    theta = np.arctan(pT/pz)
    truth_in = np.stack([phi, theta, energy], axis=1)
    truth_in = scaler.fit_transform(truth_in)
    print("Min and Max for hadrons: ", scaler.data_min_, scaler.data_max_)

    # the input 4vector is the cluster 4vector
    input_4vec = c[[1, 2, 3, 4]][selections].values
    input_4vec = scaler.fit_transform(input_4vec)
    print("Min and Max for clusters: ", scaler.data_min_, scaler.data_max_)

    shuffle(truth_in)
    shuffle(input_4vec)
    print(truth_in.shape, input_4vec.shape)


    # Split the data into training and testing
    # <HACK, FIXME, NOTE>
    # <HACK, For now a maximum of 10,000 events are used for testing, xju>
    num_test_evts = int(input_4vec.shape[0]*testing_frac)
    if num_test_evts < 10_000: num_test_evts = 10_000

    test_in, train_in = input_4vec[:num_test_evts], input_4vec[num_test_evts:max_evts]
    test_truth, train_truth = truth_in[:num_test_evts], truth_in[num_test_evts:max_evts]

    xlabels = ['phi', 'theta', 'energy']

    return (train_in, train_truth, test_in, test_truth, xlabels)



def dimuon_inclusive(filename, max_evts=1000000, testing_frac=0.1):

    df = read_dataframe(filename, " ", None)
    df=df[:-1]
    truth_data_1 = df.to_numpy().astype(np.float32)
    full_data=df.to_numpy().astype(np.float32)
    print(f"reading dimuon {df.shape[0]} events from file {filename}")

    truth_data_1 = truth_data_1[truth_data_1[:, 0] < 1000]
    full_data = full_data[full_data[:, 0] < 1000]

    print('Max pt lead value:', max(truth_data_1[:, 0]))
    scaler = MinMaxScaler(feature_range=(-1,1))
    truth_data = scaler.fit_transform(truth_data_1)
    
    shuffle(truth_data)
    shuffle(truth_data_1)

    truth_data=truth_data[:max_evts]
    truth_data_1=truth_data_1[:max_evts]

    num_test_evts = int(truth_data.shape[0]*testing_frac)
    #if num_test_evts > max_evts: num_test_evts = max_evts


    test_truth, train_truth = truth_data[:num_test_evts], truth_data[num_test_evts:max_evts]
    test_truth_1, train_truth_1 = truth_data_1[:num_test_evts], truth_data_1[num_test_evts:max_evts]


    xlabels = ['leading Muon {}'.format(name) for name in ['pT', 'eta', 'phi']] +\
              ['subleading Muon {}'.format(name) for name in ['pT', 'eta', 'phi']]

    return (None, train_truth, None, test_truth, xlabels,test_truth_1,train_truth_1,full_data)
