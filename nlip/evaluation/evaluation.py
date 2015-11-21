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
    # we're given a tuple: matrix-vector composition
    if isinstance(arg1,tuple):
        if len(arg1) == 2:
            lf = arg1[0]
            emb = arg1[1]
            # agumented matrices
            if lf.A.shape[2] == lf.A.shape[1]+1:
                ours = np.array([1-cosine(
                    np.dot(lf.word(x[0][0]),np.hstack((emb.word(x[0][1]),[1]))),
                    np.dot(lf.word(x[1][0]),np.hstack((emb.word(x[1][1]),[1])))) for x in test])
            # standard matrices
            else:
                ours = np.array([1-cosine(
                    np.dot(lf.word(x[0][0]),emb.word(x[0][1])),
                    np.dot(lf.word(x[1][0]),emb.word(x[1][1]))) for x in test])
            return spearmanr(gold,ours)
        return TypeError("Invalid input format")
    # we're only given embeddings: do cosine similarity of vectors
    elif isinstance(arg1,Embeddings):
        ours = np.array([1-cosine(arg1.word(x[0]),arg1.word(x[1])) for x in test])
        return spearmanr(gold,ours)
    return TypeError("Invalid input format")

def meanap(relpr_s, relpr_o, verb_s, verb_o, noun, test_infile):
    with smart_open(test_infile, "r") as f:
        # SBJ/OBJ targetnoun noun relpr verb noun
        test = json.load(f)
    # extract target nouns
    target_nouns  = [t for _,t,*_ in test]
    # compose relative clauses
    relative_clauses = []
    for which, t, n1, relpr, v, n2 in test:
        if which == "SBJ":
            relative_clauses.append((t, np.dot(relpr_s.word(relpr),
                np.outer(noun.word(n1),
                         np.dot(verb_o.word(v),noun.word(n2))).flatten())))
        else:
            relative_clauses.append((t,np.dot(relpr_o.word(relpr),
                np.outer(noun.word(n1),
                         np.dot(verb_s.word(v),noun.word(n2))).flatten())))
    score = 0
    targets = set(target_nouns)
    for target in targets:
        predicted = [(t, 1-cosine(noun.word(target),v)) for t,v in relative_clauses]
        predicted.sort(key=lambda x: x[1], reverse=True)
        num_hits = 0
        for i,p in enumerate(predicted):
          if p[0] == target:
              num_hits += 1.0
              score += num_hits / (i+1.0)
    return score / len(targets)
