from mmap import mmap,MAP_SHARED,PROT_READ
from numpy import array,uint8,frombuffer,float32
from numpy.random import choice,randint
from numpy.linalg import norm
from sparse import load_npz

import os
from os import listdir
from os.path import join

from progress.bar import Bar
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

    def __init__(self, dataset='test', window=5, epoch_size=100000):

        self.root = '/datadrive/musicnet'
        self.sample_frequency = 44100

        self.window = window*self.sample_frequency
        self.size = epoch_size

        # setting paths to labels and data
        assert dataset in ['train','test'], "dataset must be 'train' or 'test'"
        self.data_path = join(self.root,dataset,'data')

        self.labels = array([ 1, 7, 41, 42, 43, 44, 61, 69, 71, 72, 74], dtype=uint8)
        self.labels_path = join(self.root,dataset,'labels')

        # initialising empty dataset
        self.records = dict()
        self.ids = []
        self.open_files = []

    def __enter__(self):
        """Load dataset upon entering with <MusicNet>: statement"""
        records = listdir(self.data_path)
        bar = Bar('Loading database', max=len(records))

        for record in records:
            if not record.endswith('.npy'): continue

            id = int(record[:-4])
            self.ids += [id]

            file_pointer = os.open(join(self.data_path,record), os.O_RDONLY)
            buffer = mmap(file_pointer, 0, MAP_SHARED, PROT_READ)
            labels = load_npz(join(self.labels_path,str(id)+'.npz'))

            self.records[id] = buffer, labels, len(buffer)/sz_float
            self.open_files.append(file_pointer)
            bar.next()

        bar.finish()
        return self

    def __exit__(self, *args):
        """Clear memory upon exit of with <MusicNet>: statement"""

        for buffer,labels,size in self.records.values():
            buffer.close()

        for file_pointer in self.open_files:
            os.close(file_pointer)

        self.records = dict()
        self.open_files = []
        collect()

    def access(self,id,s):
        """
        Args:
            id (int): MusicNet id of the requested recording
            s (int): Position of the requested data point
        Returns:
            tuple: (audio, target) where target is a binary vector indicating notes on at the center of the audio.
        """

        buffer,labels,size = self.records[id]

        data = frombuffer(buffer[s*sz_float:int(s+self.window)*sz_float], dtype=float32).copy()
        label = labels[s:s+self.window]

        data /= norm(data) + epsilon
        return data,label

    def __getitem__(self, index):
        """
        Returns random (audio,score) pair
        """

        id = choice(self.ids)
        buffer,labels,size = self.records[id]

        t = randint(0,size-self.window)
        return self.access(id,t)

    def __len__(self):
        return self.size
