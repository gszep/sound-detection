from numpy import array,uint8,float32
from numpy.random import choice,randint
from numpy.linalg import norm
from scipy.sparse import lil_matrix

import h5py,h5sparse
from os.path import join
from gc import collect

sz_float = 4 # float32 in bytes
epsilon = 10e-8 # fudge factor for normalization

# subclass of torch data
from torch.utils.data import Dataset
class MusicNet(Dataset):
    """
    MusicNet dataset for large scale audio segmentation and source separation.
    Thickstun, J. et al. 2016. Learning Features of Music from Scratch. ArXiv

    Paramters
    ---------
        dataset : 'train' | 'test'
            partition to load
        window : int
            window size in seconds of a given (data,label) pair
        epoch_size : int
            number of datum pairs per epoch
    """

    def __init__(self, window=None, epoch_size=100000):

        self.labels = array([ 1, 7, 41, 42, 43, 44, 61, 69, 71, 72, 74], dtype=uint8)
        self.sample_frequency = 44100

        self.window = int(window*self.sample_frequency) if window is not None else None
        self.size = epoch_size

        # initialising dataset
        self.data_path = '/datadrive/musicnet.h5'
        self.labels_path = '/datadrive/labels.h5'

        with h5sparse.File(self.labels_path,'r') as file:
            self.ids = list(file['sparse/matrix'].h5py_group)
        # with h5py.File(self.data_path,'r') as file:
        #     self.ids = list(file)

    def __enter__(self):
        """Open file upon entering with <MusicNet>: statement"""
        self.data = h5py.File(self.data_path,'r')
        self.labels = h5sparse.File(self.labels_path,'r')
        return self

    def __exit__(self, *args):
        """Close file and clear memory upon exit of with <MusicNet>: statement"""
        self.data.close()
        self.labels.h5f.close()

        del self.data
        del self.labels
        collect()

    def __getitem__(self, index):
        """
        Returns random window of (audio,score) pair
        """

        # select random window
        id = choice(self.ids)
        data = self.data[id]['data']
        segments = self.labels[join('sparse','matrix',id)]

        if self.window is not None :
            t = randint(0,len(data)-self.window)
            return data.value[t:t+self.window],segments[t:t+self.window].toarray().astype(uint8)

        else :
            return data.value,segments.value.toarray().astype(uint8)


    def __len__(self):
        return self.size
