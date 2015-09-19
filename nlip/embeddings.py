import numpy as np
import h5py
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from nlip.utils import smart_open
from nlip import floatX

class Embeddings():
    """
    Embeddings in dense format.

    These can be instantiated in the following ways:
        Embeddings((A, index2name))
            with an array ``A`` with one embeddings per row,
            and a list of words ``index2name``.

        Embeddings((A, index2name, index2count))
            with an array ``A`` with one embeddings per row,
            a list of words ``index2name``, a list of counts ``index2count``.

        Embeddings(filename)
            with a HDF5 file containing the datasets ``A``, ``index2name``,
            and optionally ``index2count``.

    Attributes
    ----------
    A : array_like
        Array with one embeddings per row
    shape : 2-tuple
        ``(n_words, dimensionality)``
    index2name : array_like
        List of words
    name2index : dict
        Mapping from words to word indices
    index2count : array_like
        List of raw word counts from corpus

    """

    def __init__(self, arg1=None):
        self.index2name = []
        self.name2index = {}
        self.index2count = []
        self.A = []
        self.f = None
        if isinstance(arg1, str):
            self.f = h5py.File(arg1)
            self.A = self.f["A"][:]
            self.shape = self.A.shape
            self.index2name = self.f["index2name"][:]
            self.name2index = {e:i for i,e in enumerate(self.index2name)}
            if "index2count" in self.f:
                self.index2count = self.f["index2count"][:]
        elif isinstance(arg1, tuple):
            if len(arg1) == 2:
                self.A = np.asarray(arg1[0], dtype=floatX)
                self.shape = self.A.shape
                self.index2name = arg1[1]
            elif len(arg1) == 3:
                self.A = np.asarray(arg1[0], dtype=floatX)
                self.shape = self.A.shape
                self.index2name = arg1[1]
                self.index2count = arg1[2]
                if len(self.index2name) != len(self.index2count):
                    raise ValueError("Vocabulary and counts must have the same length")
            else:
                raise TypeError("Invalid input format")
            self.name2index = {e:i for i,e in enumerate(self.index2name)}

    def __getitem__(self, index):
        return self.A[index]

    def word(self, word):
        return self.A[self.name2index[word]]

    def close(self):
        self.f.close()

    def save(self, h5_outfile):
        dt = h5py.special_dtype(vlen=str)
        with h5py.File(h5_outfile, 'w') as f:
            f.create_dataset("A", data=np.asarray(self.A,dtype=floatX))
            f.create_dataset("index2name", data=np.array(self.index2name,dtype=dt))
            f.create_dataset("index2count", data=np.asarray(self.index2count,dtype=np.uint32))
