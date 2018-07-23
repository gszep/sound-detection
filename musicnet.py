from __future__ import print_function
from subprocess import call
from mmap import mmap,MAP_SHARED,PROT_READ


import os.path
import pickle
import errno
import csv
import numpy as np
import torch

from numpy import array,uint8
from numpy.random import choice
import os
from os import listdir
from os.path import join
from gc import collect

sz_float = 4    # size of a float32
sz_int = 1    # size of int8
epsilon = 10e-8 # fudge factor for normalization

# subclass of torch data
from torch.utils.data import Dataset
class MusicNet(Dataset):
    """
    MusicNet dataset for large scale audio segmentation and source separation.
    Thickstun, J. et al. 2016. Learning Features of Music from Scratch. ArXiv

    Paramters
    ---------
        train (bool, optional): If True, creates dataset from ``train_data``,
            otherwise from ``test_data``.
        normalize (bool, optional): If true, rescale input vectors to unit norm.
        window (int, optional): Size in samples of a data point.
        epoch_size (int, optional): Designated Number of samples for an "epoch"
    """

    def __init__(self, dataset='train', normalize=True, window=16384, epoch_size=100000):

        self.root = '/datadrive/musicnet'
        self.normalize = normalize
        self.window = window
        self.size = epoch_size
        self.m = 128

        # setting paths to labels and data
        assert dataset in ['train','test'], "dataset must be 'train' or 'test'"
        self.data_path = join(self.root,dataset,'data')

        self.labels = array([ 1, 7, 41, 42, 43, 44, 61, 69, 71, 72, 74], dtype=uint8)
        self.labels_path = join(self.root,dataset,'labels')

        # initialising empty dataset
        with open(join(self.labels_path,'tree.pckl'),'r') as file:
            self.labeltree = pickle.load(file)
            self.ids = self.labeltree.keys()

        self.open_files = []
        self.records = dict()

    def __enter__(self):
        """Load dataset upon entering with <MusicNet>: statement"""
        for record in listdir(self.data_path):
            if not record.endswith('.npy'): continue

            fd = os.open(os.path.join(self.data_path, record), os.O_RDONLY)
            buff = mmap(fd, 0, MAP_SHARED, PROT_READ)

            self.records[int(record[:-4])] = (buff, len(buff)/sz_float)
            self.open_files.append(fd)

    def __exit__(self, *args):
        """Clear memory upon exit of with <MusicNet>: statement"""

        for mm in self.records.values():
            mm[0].close()

        for fd in self.open_files:
            os.close(fd)

        self.open_files = []
        self.records = dict()
        collect()

    def access(self,id,s):
        """
        Args:
            id (int): MusicNet id of the requested recording
            s (int): Position of the requested data point
        Returns:
            tuple: (audio, target) where target is a binary vector indicating notes on at the center of the audio.
        """

        x = np.frombuffer(self.records[id][0][s*sz_float:int(s+self.window)*sz_float], dtype=np.float32).copy()
        if self.normalize: x /= np.linalg.norm(x) + epsilon

        y = np.zeros((self.window,128,11),dtype=np.float32)
        for t in xrange(self.window):

            for label in self.labeltree[id][s+t]:
                inst,note,_,_,_ = label.data

                if inst != 0:
                    label_index = list(self.labels==inst).index(True)
                    y[t,note,label_index] = 1.0

        return x,y

    def __getitem__(self, index):
        """
        Returns random (audio,score) pair
        """

        id = choice(self.ids)
        s = np.random.randint(0,self.records[id][1]-self.window)
        return self.access(id,s)

    def __len__(self):
        return self.size
