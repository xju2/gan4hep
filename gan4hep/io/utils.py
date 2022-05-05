from ctypes import Union
import os
import pandas as pd
import numpy as np
import pickle
from typing import Any, Union

from sklearn.preprocessing import MinMaxScaler

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


def split_to_float(df, sep=','):
    out = df
    if type(df.iloc[0]) == str:
        out = df.str.split(sep, expand=True).astype(np.float32)
    return out


class InputScaler:
    def __init__(self, feature_range=(-1, 1)):
        self.scaler = MinMaxScaler(feature_range=feature_range)
        
    def transform(self, df, outname=None):
        out_df = self.scaler.fit_transform(df)
        if outname is not None:
            self.save(outname)
        return out_df

    def save(self, outname):
        pickle.dump(self.scaler, open(outname, 'wb'))

    def load(self, outname):
        self.scaler = pickle.load(open(outname, 'rb'))

    def dump(self):
        print("Min and Max for inputs: {",\
            ", ".join(["{:.6f}".format(x) for x in self.scaler.data_min_]),\
            ", ".join(["{:.6f}".format(x) for x in self.scaler.data_max_]), "}")


class SpatialEncoder:
    """Encoder catorical data, particle IDs,
    into a vector separated by equal distance.
    Those particles appearing more often are assigned with large value.
    """
    def __init__(self):
        self.encoder = None

    def encode(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        if type(x) == np.ndarray:
            return np.array([self.encoder[x[i]] for i in range(x.shape[0])])
        else:
            return self.encoder[x]

    def __call__(self, x, *args: Any, **kwds: Any) -> Any:
        return self.encode(x)

    def train(self, x):
        all_classes, counts = np.unique(x, return_counts=True)

        # those appearing more are assigned with larger value
        classes = all_classes[np.argsort(counts)]
        num_classes = len(classes)
        self.step = 1/(num_classes-1)
        code = np.arange(0, 1+self.step, self.step)
        self.encoder = dict([(k, v) for k, v in zip(classes, code)])
