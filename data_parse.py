import sys,os,signal
import os.path
import pickle


import numpy as np
from numpy import *
import musicnet

from scipy.io import wavfile
from os.path import join
from glob import glob

from itertools import islice, chain
from multiprocessing import Pool,cpu_count
from progress.bar import Bar

def batch(l, n):
    """Yield successive n-sized batches from array"""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def processdata(id_batch):
    datalabels=array([ 1, 7, 41, 42, 43, 44, 61, 69, 71, 72, 74], dtype=uint8)
    for id,size,labeltree in id_batch:

        print '###############################',id
        bar = Bar('Processing', max=size)
        y = np.zeros((size,128,11),dtype=np.uint8)

        for t in xrange(size):

            for label in labeltree[t]:
                inst,note,_,_,_ = label.data

                if inst != 0:
                    label_index = list(datalabels==inst).index(True)
                    y[t,note,label_index] = 1

            bar.next()
        bar.finish()
        print '\n'
        np.save(os.path.join('/datadrive/musicnet/test/labels',str(id)),y)
    return True

# initialising empty dataset
with open('/datadrive/musicnet/test/labels/tree.pckl','r') as file:
    labeltree = pickle.load(file)
    ids = labeltree.keys()

records = []
for id in ids:
    records += [ (id,len(np.load('/datadrive/musicnet/test/data/'+str(id)+'.npy')),labeltree[id]) ]

pool = Pool(cpu_count())

batch_size = len(records)/(10*cpu_count())+1
for id_batch in batch(records,batch_size) :
    #processdata(id_batch)
    pool.apply_async(processdata, args=(id_batch,))
    
# wait for threads to finish
pool.close()
pool.join()
