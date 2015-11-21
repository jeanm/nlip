import numpy as np
import h5py
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from nlip import floatX

class Embeddings():
    """
    Embeddings in dense format.

    These can be instantiated in the following ways:

        Embeddings(A, index2name [, index2count])
            with an array ``A`` with one embeddings per row,
            a list of words ``index2name``, and optionally a list of counts
            ``index2count``.

        Embeddings(filename)
            with a HDF5 file containing the datasets ``A``, ``index2name``,
            and optionally ``index2count``.

    Attributes
    ----------
    A : array_like
        Array with one embedding per row
    shape : 2-tuple
        ``(n_words, dimensionality)``
    index2name : array_like
        List of words
    name2index : dict
        Mapping from words to word indices
    index2count : array_like
        List of raw word counts from corpus

    """

    def __init__(self, arg1=None, arg2=None, arg3=None):
        self.index2name = []
        self.name2index = {}
        self.index2count = []
        self.A = []
        if isinstance(arg1, str):
            with h5py.File(arg1) as fin:
                self.A = fin["A"][:]
                self.shape = self.A.shape
                self.index2name = fin["index2name"][:]
                if isinstance(self.index2name[0], (list, np.ndarray)):
                    self.name2index = {tuple(e):i for i,e in enumerate(self.index2name)}
                else:
                    self.name2index = {e:i for i,e in enumerate(self.index2name)}
                if "index2count" in fin:
                    self.index2count = fin["index2count"][:]
        elif isinstance(arg1, (np.ndarray, list)):
            self.A = np.asarray(arg1, dtype=floatX)
            self.shape = self.A.shape
            if isinstance(arg2, (list, np.ndarray)):
                self.index2name = arg2
                if isinstance(self.index2name[0], (list, np.ndarray)):
                    self.name2index = {tuple(e):i for i,e in enumerate(self.index2name)}
                else:
                    self.name2index = {e:i for i,e in enumerate(self.index2name)}
            if isinstance(arg3, (list, np.ndarray)):
                self.index2count = arg3
                if len(self.index2name) != len(self.index2count):
                    raise ValueError("Vocabulary and counts must have the same length")
        else:
            raise TypeError("Invalid input format")

    def __getitem__(self, index):
        return self.A[index]

    def word(self, word):
        return self.A[self.name2index[word]]

    def save(self, h5_outfile):
        with h5py.File(h5_outfile, 'w') as fout:
            if len(self.A) > 0:
                fout.create_dataset("A", data=np.asarray(self.A,dtype=floatX))
            fout.create_dataset("index2count", data=np.asarray(self.index2count,dtype=np.int32))
            dt = h5py.special_dtype(vlen=str)
            fout.create_dataset("index2name", data=np.array(self.index2name,dtype=dt))
