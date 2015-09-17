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
        Embeddings((A, index2word))
            with an array ``A`` with one embeddings per row,
            and a list of words `index2word`

        Embeddings((A, index2word, index2count))
            with an array ``A`` with one embeddings per row,
            a list of words ``index2word``, a list of counts ``index2count``

        Embeddings(filename)
            with a HDF5 file containing the datasets ``A``, ``index2word``,
            and optionally ``index2count``.

    Attributes
    ----------
    A : array_like
        Array with one embeddings per row
    shape : 2-tuple
        ``(n_words, dimensionality)``
    index2word : array_like
        List of words
    word2index : dict
        Mapping from words to word indices
    index2count : array_like
        List of raw word counts from corpus

    """

    def __init__(self, arg1):
        self.index2word = []
        self.word2index = {}
        self.index2count = []
        self.A = []
        self.f = None
        if isinstance(arg1, str):
            self.f = h5py.File(arg1)
            self.A = self.f["A"][:]
            self.shape = self.A.shape
            self.index2word = self.f["index2word"][:]
            self.word2index = {e:i for i,e in enumerate(self.index2word)}
            if "index2count" in self.f:
                self.index2count = self.f["index2count"][:]
        elif isinstance(arg1, tuple):
            if len(arg1) == 2:
                self.A = np.asarray(arg1[0], dtype=floatX)
                self.shape = self.A.shape
                self.index2word = arg1[1]
            elif len(arg2) == 3:
                self.A = np.asarray(arg1[0], dtype=floatX)
                self.shape = self.A.shape
                self.index2word = arg1[1]
                self.index2count = arg1[2]
                if len(self.index2word) != len(self.index2count):
                    raise ValueError("Vocabulary and counts must have the same length")
            else:
                raise TypeError("Invalid input format")
            self.word2index = {e:i for i,e in enumerate(self.index2word)}

    def __getitem__(self, index):
        return self.A[index]

    def word(self, word):
        return self.A[self.word2index[word]]

    def close(self):
        self.f.close()

    def save(self, h5_outfile):
        dt = h5py.special_dtype(vlen=str)
        with h5py.File(h5_outfile, 'w') as f:
            f.create_dataset("A", data=np.asarray(self.A,dtype=floatX))
            f.create_dataset("index2word", data=np.array(self.index2word,dtype=dt))
            f.create_dataset("index2count", data=np.asarray(self.index2count,dtype=np.uint32))

class SparseEmbeddings():
    """
    Embeddings in Compressed Sparse Row matrix format.

    These can be instantiated in the following ways:
        SparseEmbeddings(((I,J,V), index2word))
            with a triplet ``I,J,V`` of lists containing row indices,
            column indices, and values, and a list of words `index2word`

        SparseEmbeddings(((I,J,V), index2word, index2count))
            with a triplet ``I,J,V`` of lists containing row indices,
            column indices, and values, a list of words `index2word`,
            and a list of counts ``index2count``

        SparseEmbeddings(filename)
            with a HDF5 file containing the datasets ``I``, ``J``, ``V``,
            ``index2word``, and optionally ``index2count``.

    Attributes
    ----------
    A : scipy.sparse.csr_matrix
        Compressed Sparse Row matrix with one embedding per row
    shape : 2-tuple
        ``(n_words, dimensionality)``
    index2word : array_like
        List of words
    word2index : dict
        Mapping from words to word indices
    index2count : array_like
        List of raw word counts from corpus

    """

    def __init__(self, arg1):
        self.index2word = []
        self.word2index = {}
        self.index2count = []
        self.A = []
        self.f = None
        if isinstance(arg1, str):
            self.f = h5py.File(arg1)
            self.A = csr_matrix((self.f["V"][:],(self.f["I"][:],self.f["J"][:])), dtype=floatX)
            self.shape = self.A.shape
            self.index2word = self.f["index2word"][:]
            self.word2index = {e:i for i,e in enumerate(self.index2word)}
            if "index2count" in self.f:
                self.index2count = self.f["index2count"][:]
        elif isinstance(arg1, tuple):
            if len(arg1) == 2:
                if len(arg1[0]) != 3:
                    raise TypeError("Invalid input format")
                self.A = csr_matrix((arg1[0][2],(arg1[0][0],arg1[0][1])), dtype=floatX)
                self.shape = self.A.shape
                self.index2word = arg1[1]
            elif len(arg2) == 3:
                if len(arg1[0]) != 3:
                    raise TypeError("Invalid input format")
                self.A = csr_matrix((arg1[0][2],(arg1[0][0],arg1[0][1])), dtype=floatfloatXX)
                self.shape = self.A.shape
                self.index2word = arg1[1]
                self.index2count = arg1[2]
                if len(self.index2word) != len(self.index2count):
                    raise ValueError("Vocabulary and counts must have the same length")
            else:
                raise TypeError("Invalid input format")
            self.word2index = {e:i for i,e in enumerate(self.index2word)}

    def __getitem__(self, index):
        return self.A[index]

    def word(self, word):
        return self.A[self.word2index[word]]

    def close(self):
        self.f.close()

    def save(self, h5_outfile):
        dt = h5py.special_dtype(vlen=str)
        coo = self.A.tocoo()
        with h5py.File(h5_outfile, 'w') as f:
            f.create_dataset("I", data=np.asarray(coo.row,dtype=np.uint32))
            f.create_dataset("J", data=np.asarray(coo.col,dtype=np.uint32))
            f.create_dataset("V", data=np.asarray(coo.data,dtype=floatX))
            f.create_dataset("index2word", data=np.array(self.index2word,dtype=dt))
            f.create_dataset("index2count", data=np.asarray(self.index2count,dtype=np.uint32))
