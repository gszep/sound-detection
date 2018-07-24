import sys,os,signal
import os.path
import pickle
import re


import numpy as np
from numpy import *
import musicnet

from scipy.io import wavfile
from sparse import COO,save_npz

from os.path import join
from glob import glob

from itertools import islice, chain
from multiprocessing import Pool,cpu_count
from progress.bar import Bar
from gc import collect


def batch(l, n):
    """Yield successive n-sized batches from array"""
    for i in range(0, len(l), n):
        yield l[i:i + n]

@profile
def processdata(id_batch):
    datalabels=array([ 1, 7, 41, 42, 43, 44, 61, 69, 71, 72, 74], dtype=uint8)
    id,size,labeltree = id_batch

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
    save_npz(os.path.join('/datadrive/musicnet/train/labels',str(id)),COO.from_numpy(y))

    print id,'Saved!'
    del y
    collect()

# initialising empty dataset
with open('/datadrive/musicnet/train/labels/tree.pckl','r') as file:
    print 'loading tree...'
    labeltree = pickle.load(file)
    ids = labeltree.keys()
    print 'Done!'

records = len(ids)*[None]
i = 0
for id in ids:
    with open('/datadrive/musicnet/train/data/'+str(id)+'.npy') as file:
        size = int(re.search("(?<=shape': \()([0-9]+)",file.readline()).group(0))
    records[i] = (int(id),int(size),labeltree[id].copy())
    print i,id,size
    i += 1

del labeltree,ids
collect()
#pool = Pool(cpu_count())
print 'Done!'

batch_size = len(records)/(10*cpu_count())+1
#for id_batch in records :

processdata(records[0])
    #pool.apply_async(processdata, args=(id_batch,))

# # wait for threads to finish
# pool.close()
# pool.join()
