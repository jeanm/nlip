import numpy as np
import h5py
from scipy import sparse
from nlip.utils import smart_open

class LexicalFunctions():

    def __init__(self, h5_infile=None,
            A=None, index2word=None, index2count=None,
            embeddings_infile=None, vocab_infile=None):
        self.index2word = []
        self.word2index = {}
        self.index2count = []
        self.A = []
        # load from HDF5 file
        if h5_infile:
            self.f = h5py.File(h5_infile)
            self.A = self.f["A"][:]
            self.index2word = self.f["index2word"][:]
            self.index2count = self.f["index2count"][:]
        # load from embeddings array, vocabulary, and (optionally) counts
        elif A and index2word:
            self.A = A
            self.index2word = index2word
            if index2count:
                self.index2count = index2count
        # load from plaintext & npy files
        elif embeddings_infile and vocab_infile:
            self.A = np.load(embeddings_infile, mmap_mode='r').transpose(0,2,1)
            with smart_open(vocab_infile, 'r') as f:
                for line_num,line in enumerate(f):
                    tokens = line.strip().split()
                    self.index2word.append(tokens[0])
                    if len(tokens) == 2:
                        self.index2count.append(int(tokens[1]))
        # ignore count data if values are missing
        if len(self.index2word) != len(self.index2count):
            self.index2count = []
        if len(self.index2word):
            self._build_word2index()

    def __getitem__(self, index):
        return self.A[index]

    def word(self, word):
        return self.A[self.word2index[word]]

    def close(self):
        self.f.close()

    def _build_word2index(self):
            self.word2index = {e:i for i,e in enumerate(self.index2word)}

    def save(self, h5_outfile):
        dt = h5py.special_dtype(vlen=str)
        with h5py.File(h5_outfile, 'w') as f:
            At = self.A.transpose((0,2,1))
            f.create_dataset("A", data=At)
            f.create_dataset("index2word", data=np.array(self.index2word,dtype=dt))
            f.create_dataset("index2count", data=np.asarray(self.index2count,dtype=np.uint32))

    def save_vocab(self, vocab_outfile):
        vocab_size = len(self.index2word)
        if len(self.index2count) == vocab_size:
            with smart_open(vocab_outfile, 'w') as f:
                for word_index in range(vocab_size):
                    f.write(self.index2word[word_index]+' ')
                    f.write(str(self.index2count[word_index])+'\n')
        else:
            with smart_open(vocab_outfile, 'w') as f:
                for line_num,line in enumerate(f):
                    f.write(self.index2word[word_index]+'\n')
