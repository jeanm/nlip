import numpy as np
import h5py

def smart_open(filename, mode):
    if filename.endswith('.gz'):
        import gzip
        return gzip.open(filename, mode)
    if filename.endswith('.bz2'):
        import bz2
        return bz2.open(filename, mode)
    return open(filename, mode)

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
    return Embeddings(A=vecs,index2word=words)

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
    return Embeddings(A=vecs, index2word=words)

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

def save_csr_matrix(m, filename):
    with h5py.File(filename,'w') as f:
        for name in ("data", "indices", "indptr", "shape"):
                f.create_dataset(name, data=getattr(m, name))

def load_csr_matrix(filename):
    with h5py.File(filename) as f:
        data = f["data"][:]
        indices = f["indices"][:]
        indptr = f["indptr"][:]
        shape = f["shape"][:]
    m = sparse.csr_matrix((data,indices,indptr), shape=shape)
    return m

# PTB tags to universal tagset mapping, renamed as follows for compactness:
# ADJ	J
# ADP	I
# ADV	R
# CONJ	C
# DET	D
# NOUN	N
# NUM	#
# PRON	P
# PRT	T
# VERB	V
# X	X
# .	.
# based on github.com/slavpetrov/universal-pos-tags/blob/master/en-ptb.map
_newtags = defaultdict(lambda: "X", [("!", "."), ("#", "."), ("$", "."), ("``", "."), ("''", "."), ("(", "."), (")", "."), (",", "."), ("-LRB-", "."), ("-RRB-", "."), (".", "."), (":", "."), ("?", "."), ("CC", "C"), ("CD", "#"), ("DT", "D"), ("EX", "D"), ("IN", "I"), ("IN|RP", "I"), ("JJ", "J"), ("JJR", "J"), ("JJRJR", "J"), ("JJS", "J"), ("JJ|RB", "J"), ("JJ|VBG", "J"), ("MD", "V"), ("NN", "N"), ("NNP", "N"), ("NNPS", "N"), ("NNS", "N"), ("NN|NNS", "N"), ("NN|SYM", "N"), ("NN|VBG", "N"), ("NP", "N"), ("PDT", "D"), ("POS", "T"), ("PRP", "P"), ("PRP$", "P"), ("PRP|VBP", "P"), ("T", "T"), ("RB", "R"), ("RBR", "R"), ("RBS", "R"), ("RB|RP", "R"), ("RB|VBG", "R"), ("RP", "T"), ("TO", "T"), ("VB", "V"), ("VBD", "V"), ("VBD|VBN", "V"), ("VBG", "V"), ("VBG|NN", "V"), ("VBN", "V"), ("VBP", "V"), ("VBP|TO", "V"), ("VBZ", "V"), ("VP", "V"), ("WDT", "D"), ("WP", "P"), ("WP$", "P"), ("WRB", "R")])

# discard words that do not match this pattern
_pattern = re.compile("^[\w][\w'-]*$")

def cleanup_w(x):
    return x if _pattern.search(x) else None

def cleanup_w_t(x):
    return x[0]+'|'+_newtags[x[1]] if _pattern.search(x[0]) else None

def filter_ws(words):
    return [w for w in map(cleanup_w,words) if w]

def filter_ws_ts(words, tags):
    return [x for x in map(cleanup_w_t, zip(words,tags)) if x]
