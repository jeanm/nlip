import numpy as np
import h5py
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from nlip.utils import smart_open
from nlip import floatX
from copy import deepcopy

class WordFeatureMatrix():
    """
    Word-Feature co-occurrences in Compressed Sparse Row matrix format.

    These can be instantiated in the following ways:
        WordFeatureMatrix(((I,J,V), index2name))
            with a triplet ``I,J,V`` of lists containing row indices,
            column indices, values, plus a list of words ``index2name``.

        WordFeatureMatrix(((I,J,V), index2name, index2count))
            with a triplet ``I,J,V`` of lists containing row indices,
            column indices, and values, a list of words ``index2name``,
            and a list of counts ``index2count``.

        WordFeatureMatrix(filename)
            with a HDF5 file containing the datasets ``I``, ``J``, ``V``,
            ``index2name``, and optionally ``index2count``.

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

    def __init__(self, arg1=None):
        self.index2name = []
        self.name2index = {}
        self.index2count = []
        self.A = []
        if isinstance(arg1, str):
            with h5py.File(arg1) as f:
                self.A = coo_matrix((f["V"][:],(f["I"][:],f["J"][:])), dtype=floatX)
                self.shape = self.A.shape
                self.index2name = f["index2name"][:]
                self.name2index = {e:i for i,e in enumerate(self.index2name)}
                if "index2count" in f:
                    self.index2count = f["index2count"][:]
        elif isinstance(arg1, tuple):
            if len(arg1) == 2:
                if len(arg1[0]) != 3:
                    raise TypeError("Invalid input format")
                self.A = coo_matrix((arg1[0][2],(arg1[0][0],arg1[0][1])), dtype=floatX)
                self.shape = self.A.shape
                self.index2name = arg1[1]
            elif len(arg1) == 3:
                if len(arg1[0]) != 3:
                    raise TypeError("Invalid input format")
                self.A = coo_matrix((arg1[0][2],(arg1[0][0],arg1[0][1])), dtype=floatfloatXX)
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
