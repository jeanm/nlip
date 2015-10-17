from collections import namedtuple, defaultdict
from itertools import product

def find_phrases(tags, grs):
    """Return the position of all words that are part of phrases"""
    Phrases = namedtuple("Phrases", ["an", "dn", "svo", "src", "orc"])
    phrases = Phrases(an=[], dn=[], svo=[], src=[], orc=[])
    rel_pron = ["WDT","WP"]
    vdict = defaultdict(lambda: ([],[])) # verb -> ([subjs],[objs])
    v2srp = defaultdict(list)  # verb -> subject relative pron
    v2orp = defaultdict(list)  # verb -> object relative pron
    relcl = [] # noun --acl:relcl--> verb
    verbs_with_cxcomp = set()
    for gr in grs:
        head, dep = gr[1]-1, gr[2]-1
        # decrement indices since in the gr list 0 represents ROOT and
        # everything else is incremented accordingly
        if gr[1] == 0 or gr[2] == 0:
            continue
        if gr[0] == "det":
            phrases.dn.append((dep,head))
        if gr[0] == "amod":
            phrases.an.append((dep,head))
        if gr[0] == "nsubj":
            if tags[head].startswith("VB"):
                if tags[dep].startswith("NN"):
                    vdict[head][0].append(dep)
                elif tags[dep] in rel_pron:
                    v2srp[head].append(dep)
        if gr[0] == "dobj":
            if tags[head].startswith("VB"):
                if tags[dep].startswith("NN"):
                    vdict[head][1].append(dep)
                elif tags[dep] in rel_pron:
                    v2orp[head].append(dep)
        if gr[0] == "acl:relcl":
            if tags[head].startswith("NN") and tags[dep].startswith("VB"):
                relcl.append((head,dep))
        if gr[0] == "xcomp" or gr[0] == "ccomp":
            if tags[dep].startswith("VB"):
                verbs_with_cxcomp.add(dep)
    # second pass to build svos and fill gaps with None
    for v,sos in vdict.items():
        ss,os = sos
        if len(ss) == 0:
            ss = [None]
        if len(os) == 0:
            os = [None]
        phrases.svo.extend((s,v,o) for s,o in product(ss,os))
    # second pass to build relative clauses
    for noun,verb in relcl:
        if verb in verbs_with_cxcomp:
            continue
        if verb in v2srp:
            relpr = v2srp[verb]
            # subject relative clause? check if object arc is there!
            if len(vdict[verb][1]) > 0:
                phrases.src.extend(product([noun],relpr,[verb],vdict[verb][1]))
        if verb in v2orp:
            relpr = v2orp[verb]
            # object relative clause? check if subject arc is there!
            if len(vdict[verb][0]) > 0 and len(vdict[verb][1]) == 0:
                phrases.orc.extend(product([noun],relpr,vdict[verb][0],[verb]))
    return phrases
