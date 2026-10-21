"""Lab 02 - evaluation: held-out perplexity and top-k accuracy.

Run:
    python -m src.evaluate                          # large corpus, top-5 (default)
    python -m src.evaluate --corpus small --topk 3  # reproduces the original lab numbers
Results are saved in results/<corpus>/ so the two corpora never overwrite each other.
"""
import argparse
import json
import os

from src.pipeline import (CORPORA, DEFAULT_TOP_K, EOS, MIN_CONTEXT_COUNT, UNK, InterpolatedModel, NGramModel,
                          backoff_top_k, load_documents, split_by_document, top_k_accuracy)


def backoff_accuracy(tri, bi, uni, sentences, k=DEFAULT_TOP_K, min_count=MIN_CONTEXT_COUNT):
    hits = total = 0
    for s in sentences:
        for i, w in enumerate(s + [EOS]):
            guess = [x for x, _ in backoff_top_k(tri, bi, uni, s[:i], k, min_count)[0]]
            total += 1
            hits += (w if w in tri.vocab else UNK) in guess
    return hits / total


def main():
    ap = argparse.ArgumentParser(description="Evaluate the n-gram models.")
    ap.add_argument("--corpus", choices=sorted(CORPORA), default="large")
    ap.add_argument("--topk", type=int, default=DEFAULT_TOP_K, help="number of suggestions")
    args = ap.parse_args()
    K = args.topk
    out = os.path.join("results", args.corpus)

    try:
        docs = load_documents(args.corpus)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"ERROR: {e}")

    train, test = split_by_document(docs)
    n_tr = sum(len(s) for s in train); n_te = sum(len(s) for s in test)
    print(f"Corpus: {args.corpus} ({CORPORA[args.corpus]}), {len(docs)} documents")
    print(f"Split BY DOCUMENT (fixed): train = {len(train)} sentences / {n_tr} tokens, "
          f"test = {len(test)} sentences / {n_te} tokens")

    uni, bi, tri = NGramModel(train, 1, 1.0), NGramModel(train, 2, 1.0), NGramModel(train, 3, 1.0)
    tri01, tri0 = NGramModel(train, 3, 0.1), NGramModel(train, 3, 0.0)
    interp = InterpolatedModel(train, k=0.1)
    print(f"Vocabulary size V = {tri.V} (incl. </s> and <unk>)")

    rows = []
    def add(name, model):
        ppl, zero = model.perplexity(test)
        acc3, _ = top_k_accuracy(model, test, 3)
        acck, _ = top_k_accuracy(model, test, K)
        rows.append({"model": name, "perplexity": ppl, "zero_prob_share": zero,
                     "top3": acc3, f"top{K}": acck})

    add("unigram (k=1)", uni)
    add("bigram (k=1)", bi)
    add("trigram (k=1)", tri)
    add("trigram (k=0.1)", tri01)
    add("trigram unsmoothed (k=0)", tri0)
    add("interpolated 3/2/1-gram (k=0.1)", interp)

    print(f"\nmodel                               perplexity   zero-prob tokens   top-3 acc.   top-{K} acc.")
    for r in rows:
        ppl = "inf" if r["perplexity"] == float("inf") else f"{r['perplexity']:.2f}"
        print(f"{r['model']:<36}{ppl:>10}{r['zero_prob_share']:>16.1%}{r['top3']:>13.1%}{r[f'top{K}']:>13.1%}")

    # --- hard back-off: effect of min_count (how often a context must be seen to be trusted)
    b_tri, b_bi, b_uni = NGramModel(train, 3, 0.1), NGramModel(train, 2, 0.1), NGramModel(train, 1, 0.1)
    min_counts = [1, 5, 10, 20, 50]
    bo_sweep = {mc: backoff_accuracy(b_tri, b_bi, b_uni, test, K, mc) for mc in min_counts}
    bo = bo_sweep.get(MIN_CONTEXT_COUNT) or backoff_accuracy(b_tri, b_bi, b_uni, test, K)
    print(f"\nHard back-off (3->2->1-gram, k=0.1), top-{K} accuracy vs min_count "
          f"(1 = back off only if the context was never seen):")
    print("  " + ", ".join(f"min_count={mc}: {a:.1%}" for mc, a in bo_sweep.items()))
    print(f"  default min_count={MIN_CONTEXT_COUNT} -> {bo:.1%}")

    # --- effect of the smoothing constant k
    ks = [0.001, 0.01, 0.1, 0.5, 1, 2, 5]
    sweep = {n: [NGramModel(train, n, k).perplexity(test)[0] for k in ks] for n in (2, 3)}
    print("\nPerplexity vs k:")
    for n in (2, 3):
        print(f"  {n}-gram: " + ", ".join(f"k={k}: {p:.2f}" for k, p in zip(ks, sweep[n])))

    # --- failure analysis
    t = NGramModel(train, 3, 0.1)
    _, misses = top_k_accuracy(t, test, K)
    unseen_ctx = []
    for s in test:
        for i in range(len(s) + 1):
            if not t.seen_context(s[:i]):
                unseen_ctx.append(" ".join(s[max(0, i - 2):i]))
    os.makedirs(out, exist_ok=True)
    lines = [f"Trigram (k=0.1) top-{K} misses (context -> true word | suggestions):"]
    lines += [f"  '{c}' -> {w!r} | {g}" for c, w, g in misses[:6]]
    lines += [f"\nTest positions whose trigram context never appears in training: {len(unseen_ctx)}",
              f"  examples: {sorted(set(unseen_ctx))[:5]}"]
    with open(os.path.join(out, "failures.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines[:5]))
    print(lines[-2]); print(lines[-1])

    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump({"corpus": args.corpus, "top_k": K, "rows": rows, f"backoff_top{K}": bo,
                   "backoff_min_count_sweep": {str(m): a for m, a in bo_sweep.items()},
                   "k_sweep": {"k": ks, **{str(n): sweep[n] for n in sweep}},
                   "vocab_size": tri.V, "train_tokens": n_tr, "test_tokens": n_te,
                   "unseen_trigram_contexts": len(unseen_ctx)},
                  f, indent=2, default=str)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for n, c in ((2, "#8a8a8a"), (3, "#333333")):
        axes[0].plot(ks, sweep[n], marker="o", color=c, label=f"{n}-gram")
    axes[0].set_xscale("log"); axes[0].set_xlabel("smoothing constant k (log scale)")
    axes[0].set_ylabel("test perplexity (lower = better)"); axes[0].legend()
    axes[0].set_title(f"Smoothing ({args.corpus} corpus)")
    shown = [r for r in rows if r["perplexity"] != float("inf")]
    names = [r["model"].replace(" (", "\n(") for r in shown]
    y = range(len(shown))
    axes[1].barh([i - 0.2 for i in y], [r["top3"] for r in shown], height=0.4, color="#8a8a8a", label="top-3")
    axes[1].barh([i + 0.2 for i in y], [r[f"top{K}"] for r in shown], height=0.4, color="#333333", label=f"top-{K}")
    axes[1].set_yticks(list(y)); axes[1].set_yticklabels(names); axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1); axes[1].set_xlabel("accuracy (held-out documents)"); axes[1].legend(loc="lower right")
    axes[1].tick_params(axis="y", labelsize=7); axes[1].set_title("Next-word suggestion quality")
    fig.tight_layout(); fig.savefig(os.path.join(out, "ngram_results.png"), dpi=160)
    print(f"\nSaved {out}/metrics.json, failures.txt, ngram_results.png")


if __name__ == "__main__":
    main()
