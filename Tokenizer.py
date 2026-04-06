class BasicTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
    
    def _get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def _merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and \
                    ids[i] == pair[0] and \
                    ids[i+1] == pair[1]:
                newids.append(idx)
                i+=2
            else:
                newids.append(ids[i])
                i+=1
        return newids

    def train(self, text, vocab_size, verbose=False):
        ids = list(text.encode("utf-8"))
        num_merges = vocab_size - 256
        for i in range(num_merges):
            idx = 256 + i
            stats = self._get_stats(ids)
            pair = max(stats, key=stats.get)
            ids = self._merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

            if verbose:
                token_bytes = self.vocab[pair[0]] + self.vocab[pair[1]]
                token_str = token_bytes.decode("utf-8")
                print(f"{pair} -> {idx} | '{token_str}'")

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self._get_stats(tokens)
            pair = min(stats, key=lambda p: \
                       self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self._merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        tokens = b''.join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text