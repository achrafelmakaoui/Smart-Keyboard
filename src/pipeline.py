"""Lab 02 - Smart Keyboard V1: n-gram language model with add-k smoothing (no UI code here).

P(w_i | w_{i-n+1} ... w_{i-1}) = (count(context + w) + k) / (count(context) + k * V)
"""
import csv
import math
from collections import Counter, defaultdict

BOS, EOS, UNK = "<s>", "</s>", "<unk>"

# Number of suggestions shown by the keyboard (the lab started with 3; we now use 5).
DEFAULT_TOP_K = 5

# Back-off only trusts a context seen at least this many times in training
# (1 = original lab rule; 10 was best on both corpora, see evaluate.py).
MIN_CONTEXT_COUNT = 10

# Available corpora (generate them with:  python data/make_data.py --size both)
CORPORA = {
    "small": "data/corpus.csv",          # original lab corpus: 1,200 sentences, 61 word types
    "large": "data/corpus_large.csv",    # extended grammar:   6,000 sentences, 181 word types
}


# ----------------------------------------------------------------- data
def load_documents(path="data/corpus.csv"):
    """Return {doc_id: [sentence_tokens, ...]} (tokens are already lower-case, space separated).

    `path` can be a file path or a corpus name from CORPORA ("small" / "large").
    Raises a clear error (instead of a raw traceback) if the file is missing or malformed.
    """
    path = CORPORA.get(path, path)
    try:
        f = open(path, encoding="utf-8")
    except FileNotFoundError:
        size = next((name for name, p in CORPORA.items() if p == path), "both")
        raise FileNotFoundError(
            f"Corpus file '{path}' not found. Generate it first with:\n"
            f"    python data/make_data.py --size {size}") from None

    docs = defaultdict(list)
    with f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"doc_id", "sentence"} <= set(reader.fieldnames):
            raise ValueError(f"'{path}' must have the columns doc_id and sentence "
                             f"(found: {reader.fieldnames}). Regenerate it with data/make_data.py.")
        for line_no, row in enumerate(reader, start=2):
            try:
                doc_id = int(row["doc_id"])
            except (TypeError, ValueError):
                raise ValueError(f"{path}, line {line_no}: doc_id must be an integer, got {row['doc_id']!r}") from None
            words = (row["sentence"] or "").lower().split()
            if words:                                   # skip empty sentences silently
                docs[doc_id].append(words)
    if not docs:
        raise ValueError(f"'{path}' contains no sentences.")
    return dict(docs)


def split_by_document(docs, test_fraction=0.2):
    """Fixed, reproducible split BY DOCUMENT (the last 20 % of documents are the test set)."""
    ids = sorted(docs)
    cut = int(len(ids) * (1 - test_fraction))
    train = [s for i in ids[:cut] for s in docs[i]]
    test = [s for i in ids[cut:] for s in docs[i]]
    return train, test


# ----------------------------------------------------------------- model
class NGramModel:
    def __init__(self, sentences, n=3, k=1.0):
        self.n, self.k = n, k
        self.vocab = {w for s in sentences for w in s} | {EOS, UNK}   # BOS is never predicted
        self.V = len(self.vocab)
        self.ctx_counts = defaultdict(Counter)       # context tuple -> Counter(next word)
        for s in sentences:
            padded = [BOS] * (n - 1) + s + [EOS]
            for i in range(n - 1, len(padded)):
                self.ctx_counts[tuple(padded[i - n + 1:i])][padded[i]] += 1
        self.ctx_totals = {c: sum(cnt.values()) for c, cnt in self.ctx_counts.items()}

    # -- helpers
    def clean(self, words):
        """Map out-of-vocabulary words to <unk> (so the model never crashes on new words)."""
        return [w if w in self.vocab else UNK for w in words]

    def context(self, words):
        words = [BOS] * (self.n - 1) + self.clean(words)
        return tuple(words[-(self.n - 1):]) if self.n > 1 else ()

    def prob(self, ctx, word):
        counts = self.ctx_counts.get(ctx)
        total = self.ctx_totals.get(ctx, 0)
        num = (counts[word] if counts else 0) + self.k
        den = total + self.k * self.V
        return num / den if den > 0 else 0.0         # k = 0 and unseen context -> probability 0

    def next_word_distribution(self, words):
        ctx = self.context(words)
        return {w: self.prob(ctx, w) for w in self.vocab if w != UNK}

    def top_k(self, words, k=DEFAULT_TOP_K):
        dist = self.next_word_distribution(words)
        return sorted(dist.items(), key=lambda x: (-x[1], x[0]))[:k]

    def seen_context(self, words):
        return self.context(words) in self.ctx_counts

    # -- evaluation
    def perplexity(self, sentences):
        """exp( - mean log P(token) ) over every token of every sentence (EOS included).
        Returns (perplexity, share_of_zero_probability_tokens)."""
        log_sum, n_tok, zeros = 0.0, 0, 0
        for s in sentences:
            words = self.clean(s)
            padded = [BOS] * (self.n - 1) + words + [EOS]
            for i in range(self.n - 1, len(padded)):
                p = self.prob(tuple(padded[i - self.n + 1:i]), padded[i])
                n_tok += 1
                if p <= 0:
                    zeros += 1
                else:
                    log_sum += math.log(p)
        if zeros:
            return float("inf"), zeros / n_tok
        return math.exp(-log_sum / n_tok), 0.0


class InterpolatedModel:
    """p = l3 * p_trigram + l2 * p_bigram + l1 * p_unigram  (a smoother alternative to hard back-off)."""

    def __init__(self, sentences, k=0.1, lambdas=(0.6, 0.3, 0.1)):
        self.models = [NGramModel(sentences, n, k) for n in (3, 2, 1)]
        self.lambdas = lambdas
        self.vocab, self.V, self.n = self.models[0].vocab, self.models[0].V, 3

    def clean(self, words):
        return self.models[0].clean(words)

    def prob_words(self, history, word):
        history = [BOS, BOS] + history
        p = 0.0
        for lam, m in zip(self.lambdas, self.models):
            ctx = tuple(history[len(history) - (m.n - 1):]) if m.n > 1 else ()
            p += lam * m.prob(ctx, word)
        return p

    def top_k(self, words, k=DEFAULT_TOP_K):
        hist = self.clean(words)
        dist = {w: self.prob_words(hist, w) for w in self.vocab if w != UNK}
        return sorted(dist.items(), key=lambda x: (-x[1], x[0]))[:k]

    def perplexity(self, sentences):
        log_sum, n_tok = 0.0, 0
        for s in sentences:
            words = self.clean(s)
            for i, w in enumerate(words + [EOS]):
                log_sum += math.log(self.prob_words(words[:i], w))
                n_tok += 1
        return math.exp(-log_sum / n_tok), 0.0


def context_count(model, words):
    """How many times the model's context for `words` was seen in training (0 = never)."""
    return model.ctx_totals.get(model.context(words), 0)


def backoff_top_k(tri, bi, uni, words, k=DEFAULT_TOP_K, min_count=MIN_CONTEXT_COUNT):
    """Hard back-off for suggestions, with two improvements over the original version.

    1. "Seen" is not "seen enough": we only trust a context if it appeared at least
       `min_count` times in training; otherwise we back off to the next lower order.
       Example: 'students eat' appears only 2 times (always followed by 'dinner'),
       while 'eat' appears ~500 times -> the bigram knows much more.
       min_count=1 gives back the original behaviour (back off only if never seen).
    2. Ties are broken by the LOWER-order models instead of alphabetically, so that
       all the words with the same tiny smoothed probability are ranked sensibly
       (before: '.', '</s>' and '?' won ties because they sort first).
    Returns (suggestions, name_of_the_order_used).
    """
    chain = [tri, bi, uni]
    for i, model in enumerate(chain):
        if model.n == 1 or context_count(model, words) >= min_count:
            lower = chain[i + 1:]
            ctxs = [m.context(words) for m in [model] + lower]

            def score(w):   # compare by this model first, then by each lower order
                return tuple(m.prob(c, w) for m, c in zip([model] + lower, ctxs))

            # sorted() is stable: words with exactly equal scores keep alphabetical order (last resort)
            ranked = sorted(sorted(w for w in model.vocab if w != UNK), key=score, reverse=True)
            return [(w, model.prob(ctxs[0], w)) for w in ranked[:k]], f"{model.n}-gram"
    return [], "none"


def top_k_accuracy(model, sentences, k=DEFAULT_TOP_K):
    """Share of positions where the true next word is among the k suggestions."""
    hits = total = 0
    failures = []
    for s in sentences:
        for i, true_w in enumerate(s + [EOS]):
            guess = [w for w, _ in model.top_k(s[:i], k)]
            truth = true_w if true_w in model.vocab else UNK
            total += 1
            if truth in guess:
                hits += 1
            elif len(failures) < 200:
                failures.append((" ".join(s[:i]) or "<start>", true_w, guess))
    return hits / total, failures


if __name__ == "__main__":
    docs = load_documents()
    train, test = split_by_document(docs)
    m = NGramModel(train, n=3, k=0.1)
    print("after 'the girls':", m.top_k(["the", "girls"]))
    print("test perplexity  :", round(m.perplexity(test)[0], 2))
