# Idea from https://github.com/huggingface/transformers/tree/main/examples/research_projects/codeparrot
from datasketch import MinHash, MinHashLSH, LeanMinHash
from multiprocessing import Manager
from collections import defaultdict
from itertools import chain

HASH_PERMS = 256

def hash_tokens(tokens, num_perm=HASH_PERMS):
    m = MinHash(num_perm=num_perm)
    for t in tokens:
        m.update(t.encode())
    return m

def hash_examples(examples, idxs, queue):
    hashes = []
    for idx, ex in zip(idxs, examples["tokens"]):
        min_hash = hash_tokens(ex)
        lean_hash = LeanMinHash(min_hash)
        hashes.append((idx, lean_hash))
    queue.put(hashes)

def index_hashes(examples, index, dup_store):
    for example in examples:
        idx, min_hash = example
        key = idx
        if key in index.keys:
            continue
        process_duplicates(min_hash, key, dup_store, index)
        index.insert(key, min_hash)

def process_duplicates(min_hash, key, dup_store, index):
    close_duplicates = index.query(min_hash)
    # Assign input hash to at most one duplicate cluster
    if len(close_duplicates) > 0:
        for base_duplicate in close_duplicates:
            if base_duplicate in dup_store:
                dup_store[base_duplicate].add(key)
                break
        else:
            dup_store[close_duplicates[0]].add(key)

def find_extremes(cluster, data, thresh=.9):
    extremes = set()
    for dup in cluster:
        for elem in extremes:
            d1 = set(data[dup]["tokens"])
            d2 = set(data[elem]["tokens"])
            sim = len(d1 & d2) / len(d1 | d2)
            if sim > thresh:
                break
        else:
            extremes.add(dup)
    return extremes

class Deduplicator:
    def __init__(self, threshold=.9, num_perm=HASH_PERMS, processes=None):
        self.index = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.dup_store = defaultdict(set)
        self.processes = processes

    def deduplicate(self, data):
        manager = Manager()
        hash_queue = manager.Queue()
        data.map(lambda xs, idxs: hash_examples(xs, idxs, hash_queue), batched=True, with_indices=True, num_proc=self.processes, desc="Hashing")
        # Used to end the processing later on
        hash_queue.put(None)

        # Add hashes to queue
        while True:
            examples = hash_queue.get()
            if examples is None:
                break
            index_hashes(examples, self.index, self.dup_store)

        duplicate_indices = set(chain.from_iterable(self.dup_store.values()))
        extremes = self.get_extremes(data)
        duplicate_indices = duplicate_indices - extremes
        filtered = data.filter(lambda x, idx: idx not in duplicate_indices, with_indices=True, num_proc=self.processes, desc="Filtering")
        return filtered

    def get_extremes(self, data):
        extremes = map(lambda x: find_extremes(x, data), self.dup_store.values())
        extremes = set(chain.from_iterable(extremes))
        return extremes