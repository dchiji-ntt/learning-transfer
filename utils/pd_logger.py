import pandas as pd
import numpy as np
import pickle
import os

class PDLogger(object):
    def __init__(self, filename=None):
        if filename is not None:
            with open(filename, "rb") as f:
                self.dfs = pickle.load(f)
        else:
            self.dfs = dict()

        self.filename = filename

    def set_filename(self, filename):
        self.filename = filename

    def save(self):
        if self.filename is None:
            raise Exception
        else:
            with open(self.filename, "wb") as f:
                pickle.dump(self.dfs, f)

    def load(self):
        if self.filename is None:
            raise Exception
        else:
            if os.path.getsize(self.filename) <= 0:
                raise Exception
            with open(self.filename, "rb") as f:
                self.dfs = pickle.load(f)

    def add(self, attr, indices=[], values=[], columns=['results']):
        assert type(indices) is list
        assert type(values) is list
        assert len(indices) == len(values)
        assert len(indices) > 0

        if attr in self.dfs:
            df = self.dfs[attr]
            for idx in indices:
                if idx in df.index:
                    print(f'[PDLogger] Warning: The results are already set at index={idx}.')
                    return

            if columns is None:
                columns = df.columns
            df_new = pd.DataFrame(values, index=indices, columns=columns)
            self.dfs[attr] = pd.concat([df, df_new])
        else:
            self.dfs[attr] = pd.DataFrame(values, index=indices, columns=columns)

    def get_df(self, attr):
        return self.dfs[attr]
