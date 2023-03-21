import numpy as np

from gan4hep.io import GAN_INPUT_DATA_TYPE
from gan4hep.io.utils import shuffle
from gan4hep.io.utils import read_dataframe

from gan4hep.preprocessing import InputScaler

def read(filename, max_evts=None, testing_frac=0.1) -> GAN_INPUT_DATA_TYPE:
    
    df = read_dataframe(filename, " ", None)
    truth_data = df.to_numpy().astype(np.float32)

    # Remove FSR columns
    truth_data = truth_data[:, 0:6]
    # Remove high pT muons leading muons pT > 1 TeV
    # subleading muon pT > 800 GeV.
    truth_data = truth_data[truth_data[:, 0] < 1000]
    truth_data = truth_data[truth_data[:, 3] < 800]

    print(f"reading dimuon {df.shape[0]} events from file {filename}")

    scaler = InputScaler()
    truth_data = scaler.transform(truth_data, outname="dimuon_inclusive_scalar.pkl")

    shuffle(truth_data)

    num_test_evts = int(truth_data.shape[0]*testing_frac)
    if num_test_evts > 100_000: num_test_evts = 100_000


    test_truth, train_truth = truth_data[:num_test_evts], truth_data[num_test_evts:max_evts]

    xlabels = ['leading Muon {}'.format(name) for name in ['pT', 'eta', 'phi']] + \
              ['subleading Muon {}'.format(name) for name in ['pT', 'eta', 'phi']]
    
    return (None, train_truth, None, test_truth, xlabels)