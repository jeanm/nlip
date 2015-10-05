import re
from collections import defaultdict

# PTB tags to universal tagset mapping, renamed as follows for compactness:
# ADJ	A
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
_newtags = defaultdict(lambda: "X", [("!", "."), ("#", "."), ("$", "."), ("``", "."), ("''", "."), ("(", "."), (")", "."), (",", "."), ("-LRB-", "."), ("-RRB-", "."), (".", "."), (":", "."), ("?", "."), ("CC", "C"), ("CD", "#"), ("DT", "D"), ("EX", "D"), ("IN", "I"), ("IN|RP", "I"), ("JJ", "A"), ("JJR", "A"), ("JJRJR", "A"), ("JJS", "A"), ("JJ|RB", "A"), ("JJ|VBG", "A"), ("MD", "V"), ("NN", "N"), ("NNP", "N"), ("NNPS", "N"), ("NNS", "N"), ("NN|NNS", "N"), ("NN|SYM", "N"), ("NN|VBG", "N"), ("NP", "N"), ("PDT", "D"), ("POS", "T"), ("PRP", "P"), ("PRP$", "P"), ("PRP|VBP", "P"), ("T", "T"), ("RB", "R"), ("RBR", "R"), ("RBS", "R"), ("RB|RP", "R"), ("RB|VBG", "R"), ("RP", "T"), ("TO", "T"), ("VB", "V"), ("VBD", "V"), ("VBD|VBN", "V"), ("VBG", "V"), ("VBG|NN", "V"), ("VBN", "V"), ("VBP", "V"), ("VBP|TO", "V"), ("VBZ", "V"), ("VP", "V"), ("WDT", "D"), ("WP", "P"), ("WP$", "P"), ("WRB", "R")])

# discard words that do not match this pattern
_pattern = re.compile("^[\w][\w'-]*$")

def cleanup_w(x):
    return x.lower().replace("-","") if _pattern.search(x) else None

def cleanup_w_t(x):
    return x[0].lower().replace("-","")+'|'+_newtags[x[1]] if _pattern.search(x[0]) else None

def filter_ws(words):
    return [w for w in map(cleanup_w,words) if w]

def filter_ws_ts(words, tags):
    return [x for x in map(cleanup_w_t, zip(words,tags)) if x]

def find_an(tags, grs):
    """Return the position of all adjective and nouns that are part of ANs"""
    anlist = []
    for gr in grs:
        if gr[0] == "amod" and gr[1] != 0 and gr[2] != 0:
            # decrement indices since in the gr list 0 represents ROOT and
            # everything else is incremented accordingly

            anlist.append((gr[2]-1,gr[1]-1))
    return anlist

def find_det(tags, grs):
    """Return the position of all determiners and nouns that are part of NPs"""
    detlist = []
    for gr in grs:
        if gr[0] == "det" and gr[1] != 0 and gr[2] != 0:
            # decrement indices since in the gr list 0 represents ROOT and
            # everything else is incremented accordingly
            detlist.append((gr[2]-1,gr[1]-1))
    return detlist

def get_window(index, length, k=2):
    """Return a list of tuples (pos, dist_from_centre) for index's window"""
    start_win = max(0, index - k)
    end_win = min(length - 1, index + k)
    return sorted(((pos,abs(index-pos)) for pos in range(start_win, end_win+1)
            if pos != index), key=lambda x: x[1])

def get_phrase_window(indices, length, k=2):
    window = [k+1] * length  # maps position in sentence to smallest distance from index word
    indices = set(indices)  # used to make sure I don't include indices in the window
    for index in indices:
        for pos, distance in get_window(index, length, k):
            if distance < window[pos]:  # only update distance if lower than what we already found
                window[pos] = distance
    # positions are sorted by smallest distance from any index word, and
    # do not include index words themselves
    return sorted((tuple(element) for element in enumerate(window)
            if element[1] <= k and element[0] not in indices),
            key=lambda x: x[1])
