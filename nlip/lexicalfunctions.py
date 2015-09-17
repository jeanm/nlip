import numpy as np
import h5py
from scipy import sparse
from nlip.utils import smart_open

class LexicalFunctions():
    """
    Matrix Lexical Functions.

    These can be instantiated in the following ways:
        LexicalFunctions((A, index2word))
            with a third-order array ``A`` (the first dimension indexing
            functions), and a list of function names ``index2word``.

        LexicalFunctions((A, index2word, index2count))
            with a third-order array ``A`` (the first dimension indexing
            functions), a list of function names ``index2word``, and a list of
            counts ``index2count``.

        LexicalFunctions(filename)
            with a HDF5 file containing the datasets ``A``, ``index2word``,
            and optionally ``index2count``.

    Attributes
    ----------
    A : array_like
        Third-order array with the first dimension indexing functions
    shape : 3-tuple
        ``(n_functions, function_rows, function_cols)``
    index2word : array_like
        List of words (names of functions)
    word2index : dict
        Mapping from words to function indices
    index2count : array_like
        List of raw word counts from corpus
    index2nargs : array_like
        List of number of arguments of a function from corpus

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
