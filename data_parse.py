from numpy import *
import musicnet

from scipy.sparse import lil_matrix,save_npz
import pickle
import re

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


def processdata(id):
    dset = 'test'
    datalabels=array([ 1, 7, 41, 42, 43, 44, 61, 69, 71, 72, 74], dtype=uint8)

    with open('/datadrive/musicnet/'+dset+'/data/'+str(id)+'.npy','r') as file:
        size = int(re.search("(?<=shape': \()([0-9]+)",file.readline()).group(0))

    with open('/datadrive/musicnet/'+dset+'/labels/'+str(id)+'.pckl','r') as file:
        labeltree = pickle.load(file)

    bar = Bar('Processing '+str(id), max=size)
    y = lil_matrix((size,11),dtype=uint8)

    for t in xrange(size):

        for label in labeltree[t]:

            inst,note,_,_,_ = label.data
            label_index = list(datalabels==inst).index(True)
            if inst != 0: y[t,label_index] = note

        bar.next()

    bar.finish()
    print 'Saving',id,'...'
    save_npz(join('/datadrive/musicnet/'+dset+'/labels',str(id)),y.tocsr())
    print id,'Saved!'
    del y
    collect()

dset = 'test'
def main():

    ids = [ int(x[-8:-4]) for x in glob('/datadrive/musicnet/'+dset+'/data/*.npy')]
    saved = [ int(x[-8:-4]) for x in glob('/datadrive/musicnet/'+dset+'/labels/*.npz')]

    pool = Pool(cpu_count())
    for id in ids :
        if id not in saved :
            pool.apply_async(processdata, args=(id,))

    # wait for threads to finish
    pool.close()
    pool.join()

main()
