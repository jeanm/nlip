import re
from collections import defaultdict

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
