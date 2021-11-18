import pandas as pd
import numpy as np


def shuffle(array: np.ndarray):
    from numpy.random import MT19937
    from numpy.random import RandomState, SeedSequence
    np_rs = RandomState(MT19937(SeedSequence(123456789)))
    np_rs.shuffle(array)


def herwig_angles(filename,
        max_evts=None, testing_frac=0.1):
    """
    This reads the Herwig dataset where one cluster decays
    into two particles.
    In this case, we ask the GAN to predict the theta and phi
    angle of one of the particles
    """
    if type(filename) == list:
        print(filename)
        df_list = [
            pd.read_csv(f, sep=';', header=None, names=None, engine='python')
                for f in filename
        ]
        df = pd.concat(df_list, ignore_index=True)
        filename = filename[0]
    else:
        df = pd.read_csv(filename, sep=';', 
                    header=None, names=None, engine='python')

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

    return (train_in, train_truth, test_in, test_truth)


def dimuon_inclusive(filename, max_evts=None, testing_frac=0.1):
    df = pd.read_csv(filename, sep=' ', header=None, names=None)
    truth_data = df.to_numpy()
    shuffle(truth_data)

    num_test_evts = int(truth_data.shape[0]*testing_frac)
    if num_test_evts > 10_000: num_test_evts = 10_000
    
    test_truth, train_truth = truth_data[:num_test_evts], truth_data[num_test_evts:max_evts]
    return (None, train_truth, None, test_truth)