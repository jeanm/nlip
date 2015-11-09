from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import json
import numpy as np
from nlip.utils import smart_open
from nlip import Embeddings, LexicalFunctions

def similarity(arg1, test_infile):
    with smart_open(test_infile, "r") as f:
        test = json.load(f)
    gold = np.array([float(x[2]) for x in test])
    if isinstance(arg1,tuple):
        if len(arg1) == 2:
            lf = arg1[0]
            emb = arg1[1]
            if lf.A.shape[2] == lf.A.shape[1]+1:
                ours = np.array([1-cosine(
                    np.dot(lf.word(x[0][0]),np.hstack((emb.word(x[0][1]),[1]))),
                    np.dot(lf.word(x[1][0]),np.hstack((emb.word(x[1][1]),[1])))) for x in test])
            else:
                ours = np.array([1-cosine(
                    np.dot(lf.word(x[0][0]),emb.word(x[0][1])),
                    np.dot(lf.word(x[1][0]),emb.word(x[1][1]))) for x in test])
            return spearmanr(gold,ours)
        return TypeError("Invalid input format")
    elif isinstance(arg1,Embeddings):
        ours = np.array([1-cosine(arg1.word(x[0]),arg1.word(x[1])) for x in test])
        return spearmanr(gold,ours)
    return TypeError("Invalid input format")
