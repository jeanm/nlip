import numpy as np
import h5py
from scipy.sparse import coo_matrix, issparse
from sklearn.neighbors import NearestNeighbors
from nlip.utils import smart_open
from nlip import floatX
from copy import deepcopy

class WordFeatureMatrix():
    """
    Word-Feature matrix in Compressed Sparse Row format.

    These can be instantiated in the following ways:
        WordFeatureMatrix((I,J,V), index2name [, index2count])
            with a triplet ``I,J,V`` of lists containing row (word) indices,
            column (feature) indices, and values; a list of word names
            ``index2name``; and optionally a list of word counts
            ``index2count``. It is assumed that words and features coincide,
            i.e. that the matrix is square.

        WordFeatureMatrix((I,J,V), (index2name, index2name_f) [, (index2count, index2count_f)])
            with a triplet ``I,J,V`` of lists containing row (word) indices,
            column (feature) indices, and values; a pair of lists containing
            word and feature names; and optionally a pair of lists containing
            word and feature counts.

        WordFeatureMatrix(filename)
            with a HDF5 file containing the datasets ``I``, ``J``, ``V``,
            ``index2name``; and optionally ``index2count``, ``index2name_f``,
            and ``index2count_f``.

    Attributes
    ----------
    A : scipy.sparse.csr_matrix
        Compressed Sparse Row matrix with one word per row,
        one feature per column
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
        self.index2name_f = []
        self.name2index_f = {}
        self.index2count_f = {}
        self.A = []

        # we are given a filename as 1st arg
        if isinstance(arg1, str):
            with h5py.File(arg1) as f:
                self.A = coo_matrix((f["V"][:],(f["I"][:],f["J"][:])), dtype=floatX)
                self.shape = self.A.shape
                self.index2name = f["index2name"][:]
                self.name2index = {e:i for i,e in enumerate(self.index2name)}
                if "index2name_f" in f:
                    self.index2name_f = f["index2name_f"][:]
                    self.name2index_f = {e:i for i,e in enumerate(self.index2name_f)}
                else:
                    self.index2name_f = self.index2name
                    self.name2index_f = self.name2index
                if "index2count" in f:
                    self.index2count = f["index2count"][:]
                    if "index2count_f" in f:
                        self.index2count_f = f["index2count_f"][:]
                    else:
                        self.index2count_f = self.index2count
        # we are given an IJV triplet as 1st arg
        elif isinstance(arg1, tuple):
            if len(arg1[0]) != 3: # must be a triplet
                raise TypeError("First argument must be a triplet of lists")
            self.A = coo_matrix((arg1[0][2],(arg1[0][0],arg1[0][1])), dtype=floatX)
            self.shape = self.A.shape
            # second argument: list(s) of names
            if isinstance(arg2, list):
                self.index2name = arg2
                self.name2index = {e:i for i,e in enumerate(self.index2name)}
                self.index2name_f = self.index2name
                self.name2index_f = self.name2index
            elif isinstance(arg2, tuple) and len(arg2)==2:
                self.index2name = arg2[0]
                self.name2index = {e:i for i,e in enumerate(self.index2name)}
                self.index2name_f = arg2[1]
                self.name2index_f = {e:i for i,e in enumerate(self.index2name_f)}
            else:
                return TypeError("Second argument must be a list or pair of lists")
            # third argument: list(s) of counts
            if isinstance(arg3, (list, np.ndarray)):
                self.index2count = arg3
                self.index2count_f = self.index2count
            elif isinstance(arg3, tuple) and len(arg3)==2:
                self.index2count = arg3[0]
                self.index2count_f = arg3[1]
            else:
                return TypeError("Third argument must be a list or pair of lists")
        else:
            raise TypeError("Invalid input format")

    def __getitem__(self, index):
        return self.A[index]

    def word(self, word):
        return self.A[self.name2index[word]]

    def save(self, h5_outfile):
        dt = h5py.special_dtype(vlen=str)
        coo = self.A.tocoo()
        with h5py.File(h5_outfile, "w") as f:
            f.create_dataset("I", data=np.asarray(coo.row,dtype=np.uint32))
            f.create_dataset("J", data=np.asarray(coo.col,dtype=np.uint32))
            f.create_dataset("V", data=np.asarray(coo.data,dtype=floatX))
            f.create_dataset("index2name", data=np.array(self.index2name,dtype=dt))
            f.create_dataset("index2count", data=np.asarray(self.index2count,dtype=np.uint32))

    def scale(self, weights, what="rows"):
        """
        Scales rows or columns of the IJV sparse matrix.

        If scaling rows, the scaled matrix will be such that::

            V[i] = I[i] * weights[i]

        with ``J`` instead of ``I`` if scaling columns

        Parameters
        ----------
        weights : array_like
            list of weights assigned to each index
        what : str
            ``rows`` or ``columns``

        Returns
        -------
        WordFeatureMatrix
            a scaled copy of ``self``
        """
        weights = np.asarray(weights)
        if what is "rows":
            newcounts = deepcopy(self)
            newcounts.A.data = np.multiply(self.A.data,weights[self.A.row])
            return newcounts
        elif what is "cols":
            newcounts = deepcopy(self)
            newcounts.A.data = np.multiply(self.A.data,weights[self.A.col])
            return newcounts
        else:
            raise ValueError("'what' must be either 'rows' or 'cols'")
