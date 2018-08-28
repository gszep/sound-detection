from numpy import *
from numpy.random import randint
import musicnet

from scipy.sparse import lil_matrix
import h5py
import h5sparse

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
    datalabels=array([ 1, 7, 41, 42, 43, 44, 61, 69, 71, 72, 74], dtype=uint8)
    with h5py.File('/datadrive/musicnet.h5','r') as file:

        with h5sparse.File('/datadrive/labels.h5') as sparse:
            doit = join('sparse','matrix',id) not in sparse

        if doit :

            size = len(file[id]['data'])
            bar = Bar('Processing '+str(id), max=len(file[id]['labels']))

            segments = lil_matrix((size,11),dtype=uint8)
            for _,end_time,instrument,_,note,_,start_time in file[id]['labels']:
                label_index = list(datalabels==instrument).index(True)

                for t in range(start_time,end_time) :
                    segments[t,label_index] = note

                bar.next()

            with h5sparse.File('/datadrive/labels.h5') as sparse:
                sparse.create_dataset(join('sparse','matrix',id),data=segments.tocsr())

            bar.finish()
            print('Done!')

            del segments
            collect()

def main():
    ids = musicnet.MusicNet().ids

#    pool = Pool(cpu_count())
    for id in ids :
        processdata(id)
#        pool.apply_async(processdata, args=(id,))

    # wait for threads to finish
#    pool.close()
#    pool.join()

main()
