from __future__ import print_function
from subprocess import call
import os,mmap
import os.path
import pickle
import errno
import csv
import numpy as np
import torch

from intervaltree import IntervalTree
from scipy.io import wavfile

sz_float = 4    # size of a float
epsilon = 10e-8 # fudge factor for normalization

# subclass of torch data
from torch.utils.data import Dataset
class MusicNet(Dataset):
    """`MusicNet <http://homes.cs.washington.edu/~thickstn/musicnet.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset
        train (bool, optional): If True, creates dataset from ``train_data``,
            otherwise from ``test_data``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        mmap (bool, optional): If true, mmap the dataset for faster access times.
        normalize (bool, optional): If true, rescale input vectors to unit norm.
        window (int, optional): Size in samples of a data point.
        pitch_shift (int,optional): Integral pitch-shifting transformations.
        jitter (int, optional): Continuous pitch-jitter transformations.
        epoch_size (int, optional): Designated Number of samples for an "epoch"
    """

    train_data, train_labels, train_tree = 'train/data', 'train/labels', 'tree.pckl'
    test_data, test_labels, test_tree = 'test/data', 'test/labels', 'tree.pckl'

    def __init__(self, train=True, mmap=True, normalize=True, window=16384, pitch_shift=0, jitter=0., epoch_size=100000):

        self.root = '/datadrive/musicnet'
        self.mmap = mmap
        self.normalize = normalize
        self.window = window
        self.pitch_shift = pitch_shift
        self.jitter = jitter
        self.size = epoch_size
        self.m = 128

        self.root = os.path.expanduser(root)

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if train:
            self.data_path = os.path.join(self.root, self.train_data)
            labels_path = os.path.join(self.root, self.train_labels, self.train_tree)
        else:
            self.data_path = os.path.join(self.root, self.test_data)
            labels_path = os.path.join(self.root, self.test_labels, self.test_tree)

        with open(labels_path) as f:
            self.labels = pickle.load(f)

        self.rec_ids = self.labels.keys()
        self.records = dict()
        self.open_files = []

    def __enter__(self):
        for record in os.listdir(self.data_path):
            if not record.endswith('.npy'): continue
            if self.mmap:
                fd = os.open(os.path.join(self.data_path, record), os.O_RDONLY)
                buff = mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)
                self.records[int(record[:-4])] = (buff, len(buff)/sz_float)
                self.open_files.append(fd)
            else:
                f = open(os.path.join(self.data_path, record))
                self.records[int(record[:-4])] = (os.path.join(self.data_path, record),os.fstat(f.fileno()).st_size/sz_float)
                f.close()

    def __exit__(self, *args):
        if self.mmap:
            for mm in self.records.values():
                mm[0].close()
            for fd in self.open_files:
                os.close(fd)
            self.records = dict()
            self.open_files = []

    def access(self,rec_id,s,shift=0,jitter=0):
        """
        Args:
            rec_id (int): MusicNet id of the requested recording
            s (int): Position of the requested data point
            shift (int, optional): Integral pitch-shift data transformation
            jitter (float, optional): Continuous pitch-jitter data transformation
        Returns:
            tuple: (audio, target) where target is a binary vector indicating notes on at the center of the audio.
        """

        scale = 2.**((shift+jitter)/12.)

        if self.mmap:
            x = np.frombuffer(self.records[rec_id][0][s*sz_float:int(s+scale*self.window)*sz_float], dtype=np.float32).copy()
        else:
            fid,_ = self.records[rec_id]
            with open(fid, 'rb') as f:
                f.seek(s*sz_float, os.SEEK_SET)
                x = np.fromfile(f, dtype=np.float32, count=int(scale*self.window))

        if self.normalize: x /= np.linalg.norm(x) + epsilon

        xp = np.arange(self.window,dtype=np.float32)
        x = np.interp(scale*xp,np.arange(len(x),dtype=np.float32),x).astype(np.float32)

        y = np.zeros(self.m,dtype=np.float32)
        for label in self.labels[rec_id][s+scale*self.window/2]:
            y[label.data[1]+shift] = 1

        return x,y

    def __getitem__(self, index):
        """
        Args:
            index (int): (ignored by this dataset; a random data point is returned)
        Returns:
            tuple: (audio, target) where target is a binary vector indicating notes on at the center of the audio.
        """

        shift = 0
        if self.pitch_shift> 0:
            shift = np.random.randint(-self.pitch_shift,self.pitch_shift)

        jitter = 0.
        if self.jitter > 0:
            jitter_amount = np.random.uniform(-self.jitter,self.jitter)

        rec_id = self.rec_ids[np.random.randint(0,len(self.rec_ids))]
        s = np.random.randint(0,self.records[rec_id][1]-(2.**((shift+jitter)/12.))*self.window)
        return self.access(rec_id,s,shift,jitter)

    def __len__(self):
        return self.size

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.train_data)) and \
            os.path.exists(os.path.join(self.root, self.test_data)) and \
            os.path.exists(os.path.join(self.root, self.train_labels, self.train_tree)) and \
            os.path.exists(os.path.join(self.root, self.test_labels, self.test_tree))
