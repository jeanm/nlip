from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import json
import numpy as np
from nlip.utils import smart_open

def similarity(embeddings, test_infile):
    with smart_open(test_infile, "r") as f:
        test = json.load(f)
    gold = np.array([float(x[2]) for x in test])
    ours = np.array([1-cosine(embeddings.word(x[0]),embeddings.word(x[1])) for x in test])
    return spearmanr(gold,ours)
