from functools import partial
import random
import numpy as np

class MinHash:

    def __init__(self, n_hash):
        self.n_hash = n_hash
        self.n = 4294967311
        self.hash_functions = None
        self.words=set()
    
    def add(self, word):
        old = len(self.words)
        self.words.add(word)
        n = max(self.n, word)
        if len(self.words)>old or n>self.n:
            self.hash_functions = None
            self.n=n

    def compute_signature(self, min_set):
        if self.hash_functions is None:
            self.generate_h_functions()
        return np.array([min(map(hash_f, min_set)) for hash_f in self.hash_functions])
    
    def generate_h_functions(self):
        self.hash_functions= [lambda x: (x*random.randint(1, 100) +random.randint(1, 100))%(self.n+1) for _ in range(self.n_hash)]

    def get_indexed_buckets(self, hash_sets, r):
        ret = dict()
        for hs in range(len(hash_sets)):
            sig = hash_sets[hs].get_signature()
            for i in range(0, self.n_hash-r,r):
                if sig[i:i+r] not in ret:
                    ret[sig[i:i+r]] = [hs]
                else:
                    ret[sig[i:i+r]].append(hs)
        return ret


class MinHashSet:
    def __init__(self, mh, initial_set = set()):
        self.group = mh
        self.words = set()
        self.signature = None
        self.group_words= None
        self.group_n= None
        for w in initial_set:
            self.add(w)

    def is_obsolete_signature(self):
        return self.group_words != len(self.group.words) or self.group_n!=self.group.n

    def add(self, word):
        self.words.add(word)
        self.group.add(word)
        self.signature=None

    def get_signature(self):
        if self.signature is None or self.is_obsolete_signature():
            self.signature = self.group.compute_signature(self.words)
            self.group_n = self.group.n
            self.group_words = len(self.group.words)
        return self.signature

    def jaccard(self, other):
        other_sig = other.get_signature()
        self_sig = self.get_signature()
        return np.sum(self_sig==other_sig)/len(self_sig)

    def get_similar_documents(self, idx_doc, r):
        ret = []
        sig = self.get_signature()
        for i in range(0, self.n_hash-r,r):
            if sig[i:i+r] in idx_doc:
                ret += idx_doc[sig[i:i+r]]
        return set(ret)
        

    


def LSH(documents, n, query):
    b, r = get_parameters(n)
    if b is None:
        print(f"choose another signature lenght!")
        return None
    mh = MinHash(n)
    hashed_docs = list(map(lambda doc: MinHashSet(mh, doc), documents))
    hashed_query = MinHashSet(mh, query)
    idx = mh.get_indexed_buckets(hashed_docs, r)
    similar_docs = hashed_query.get_similar_documents(idx, r)
    return [(hashed_docs[d], hashed_query.jaccard(hashed_docs[d])) for d in similar_docs]