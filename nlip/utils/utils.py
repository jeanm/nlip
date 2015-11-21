import numpy as np
import h5py
from math import floor, pow, log10
from nlip import Embeddings

def smart_open(filename, mode):
    if filename.endswith('.gz'):
        import gzip
        return gzip.open(filename, mode)
    if filename.endswith('.bz2'):
        import bz2
        return bz2.open(filename, mode)
    return open(filename, mode)

def si(num):
    if num < 1000: return "{0: >5.1f}".format(num)
    exp = min(int(floor(log10(num)/3)),5)
    units = "kMGTP"
    return "{0:.1f}".format(num / pow(1000, exp))+units[exp-1]

def hms(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{:>02}h{:>02}m{:02.0f}s".format(h, m, s)


## Load vectors in plaintext format, return Embeddings object
def load_plaintext_vecs(filename):
    words = []
    vecs_raw = []
    with smart_open(filename, 'r') as f:
        for line in f:
            line = line.strip().lower().split()
            words.append(line[0])
            vecs_raw.append([float(n) for n in line[1:]])
    vecs = np.zeros((len(vecs_raw), len(vecs_raw[0])), dtype=np.float32)
    for i in range(len(vecs)):
        vecs[i] = np.asarray(vecs_raw[i], dtype=np.float32)
    return Embeddings(vecs,words)

## Load vectors in word2vec binary format, return Embeddings object
def load_word2vec_vecs(filename):
    with smart_open(filename, 'rb') as f:
        header = f.readline().decode()
        vocab_size, num_features = map(int, header.split())
        vecs = np.zeros((vocab_size, num_features), dtype=np.float32)
        words = []
        binary_len = np.dtype(np.float32).itemsize * num_features
        for line_no in range(vocab_size):
            word = []
            while True:
                char = f.read(1)
                if char == b' ':
                    break
                if char != b'\n':
                    word.append(char)
            vecs[line_no] = np.fromstring(f.read(binary_len), dtype=np.float32)
            words.append(b''.join(word).decode())
    return Embeddings(vecs, words)

## Load vectors in sparse format, return word list and scipy.sparse.csr_matrix
# The format is:
#
# word1
# feature value
# feature value
# ...
# word2
# feature value
# feature value
# ...
def load_sparse_vecs(filename, sep=None):
    with smart_open(filename, 'r') as f:
        # first build the vocabulary
        words = []
        for line in f:
            line = line.strip().split()
            if len(line) == 1: words.append(line[0])
    print('Found '+str(len(words))+' words')
    rows = []
    cols = []
    data = []
    with smart_open(filename, 'r') as f:
        inv = {e:i for i,e in enumerate(words)} # inverse mapping
        current = None # current word we're building
        for line_no,line in enumerate(f):
            line = line.strip().split(sep=sep)
            if len(line) == 1:
                if line[0] in inv:
                    current = inv[line[0]]
                else:
                    current = None
            if len(line) == 2 and current is not None:
                if line[0] in inv:
                    cols.append(inv[line[0]])
                    rows.append(current)
                    data.append(float(line[1]))
    return (rows,cols,data), words

def bisect_right(a, x, lo=0, hi=None):
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x > a[mid]: hi = mid
        else: lo = mid+1
    return lo

