import pickle
from sklearn.preprocessing import MinMaxScaler

# <TODO> Use different scaler methods
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
        return self

    def load(self, outname):
        self.scaler = pickle.load(open(outname, 'rb'))
        return self

    def dump(self):
        print("Min and Max for inputs: {",\
            ", ".join(["{:.6f}".format(x) for x in self.scaler.data_min_]),\
            ", ".join(["{:.6f}".format(x) for x in self.scaler.data_max_]), "}")