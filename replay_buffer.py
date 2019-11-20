import pandas as pd

"""
        A buffer that stores agent's playing history. Used to replay games to enhance learning ability.        
"""


class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = pd.DataFrame()
        self._maxsize = size

    def add(self, data_df):
        self._storage = pd.concat([self._storage, data_df])
        size_diff = len(self._storage) - self._maxsize
        if size_diff > 0:
            self._storage = self._storage.iloc[size_diff:]
        print('Storage new len: ', len(self._storage))

    def sample(self, batch_size):
        batch_size = min(len(self._storage), batch_size)
        return self._storage.sample(batch_size)
