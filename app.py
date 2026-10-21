"""
Lab 02 — Smart Keyboard V1 : Streamlit interface
Run from the project root:   python -m streamlit run app.py
(generate the corpora first:  python data/make_data.py --size both)
"""
import difflib
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.pipeline import (BOS, DEFAULT_TOP_K, EOS, MIN_CONTEXT_COUNT, UNK, NGramModel,
                          backoff_top_k, context_count, load_documents, split_by_document)

RESULTS = Path(__file__).parent / "results"
BACKOFF, INTERP = "Back-off 3→2→1", "Interpolated 3/2/1"
MODEL_NAMES = [BACKOFF, INTERP, "Trigram", "Bigram", "Unigram"]

st.set_page_config(page_title="Smart Keyboard V1", layout="wide")

# Typography. The font itself (Outfit) is set in .streamlit/config.toml; this only fine-tunes sizes,
# weights and spacing. Selectors use Streamlit's stable data-testid attributes.
st.html("""
<style>
  /* small uppercase labels above every control */
  [data-testid="stWidgetLabel"] p {
    font-size: 0.72rem; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: #6b7280;
  }
  /* page header */
  .kb-eyebrow { font-size: 0.7rem; font-weight: 500; letter-spacing: 0.14em; text-transform: uppercase;
                color: #9ca3af; margin: 0 0 0.25rem 0; }
  .kb-title   { font-size: 2.4rem; font-weight: 600; letter-spacing: -0.02em; line-height: 1.1; margin: 0; }
  .kb-sub     { font-size: 1.05rem; font-weight: 300; color: #6b7280; margin: 0.5rem 0 0.75rem 0; }
  /* the text the user types: larger, like a real input field */
  [data-testid="stTextInput"] input { font-size: 1.2rem; font-weight: 400; padding: 0.7rem 0.9rem; }
  /* suggestion and action buttons */
  [data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-secondary"] p {
    font-size: 1rem; font-weight: 500; letter-spacing: 0.01em;
  }
  /* tabs and expander titles */
  [data-baseweb="tab"] p { font-size: 0.95rem; font-weight: 500; }
  [data-testid="stExpander"] summary p { font-weight: 500; }
  /* sidebar heading and footer text */
  [data-testid="stSidebar"] h2 { font-size: 1.15rem; font-weight: 600; letter-spacing: -0.01em; }
  [data-testid="stCaptionContainer"] { font-weight: 300; }
  .kb-unknown { font-size: 0.95rem; font-weight: 400; color: #374151; padding-top: 0.45rem; }
  .kb-unknown b { font-weight: 600; }
</style>
""")


# ----------------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------------
def tokenize(text):
    """Lower-case words, with '.' and '?' as separate tokens (same convention as the corpus).
    Digits and other symbols are ignored, so any input is safe."""
    return re.findall(r"[a-z]+|[.?]", text.lower())


def did_you_mean(word, vocab, n=3):
    """Closest vocabulary words by spelling: 'breakfst' -> 'breakfast'."""
    return difflib.get_close_matches(word, [w for w in vocab if w not in (EOS, UNK, BOS)], n=n, cutoff=0.6)


# ----------------------------------------------------------------------------
# Data and models (cached per corpus)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading corpus...")
def get_split(corpus):
    docs = load_documents(corpus)
    return split_by_document(docs)


@st.cache_resource(show_spinner=False)
def get_ngram(corpus, n, k):
    train, _ = get_split(corpus)
    return NGramModel(train, n=n, k=k)


class KeyboardModel:
    """One interface for every model: top_k, word_prob, and which order was used."""

    def __init__(self, corpus, name, k, min_count):
        self.name, self.min_count = name, min_count
        self.tri, self.bi, self.uni = (get_ngram(corpus, n, k) for n in (3, 2, 1))
        self.single = {"Trigram": self.tri, "Bigram": self.bi, "Unigram": self.uni}.get(name)
        self.vocab = self.tri.vocab

    def pick(self, words):
        """Model used for this context (back-off: first order seen at least min_count times)."""
        if self.single:
            return self.single
        for m in (self.tri, self.bi, self.uni):
            if m.n == 1 or context_count(m, words) >= self.min_count:
                return m

    def word_prob(self, words, w):
        if self.name == INTERP:
            return sum(l * m.prob(m.context(words), w)
                       for l, m in zip((0.6, 0.3, 0.1), (self.tri, self.bi, self.uni)))
        m = self.pick(words)
        return m.prob(m.context(words), w)

    def top_k(self, words, k=DEFAULT_TOP_K):
        if self.name == BACKOFF:
            return backoff_top_k(self.tri, self.bi, self.uni, words, k, self.min_count)[0]
        if self.single:
            return self.single.top_k(words, k)
        dist = {w: self.word_prob(words, w) for w in self.vocab if w != UNK}
        return sorted(dist.items(), key=lambda x: (-x[1], x[0]))[:k]

    def explain(self, words):
        if self.name == INTERP:
            return "mix of 3-, 2- and 1-gram (0.6 / 0.3 / 0.1)", self.tri.context(words)
        m = self.pick(words)
        return f"{m.n}-gram", m.context(words)


@st.cache_data(show_spinner="Evaluating on the test documents (≈30 s the first time)...")
def evaluate(corpus, name, k, min_count):
    """Perplexity, top-3 and top-5 accuracy on the held-out documents."""
    model = KeyboardModel(corpus, name, k, min_count)
    _, test = get_split(corpus)
    log_sum = n = zeros = hit3 = hit5 = 0
    for s in test:
        for i, w in enumerate(s + [EOS]):
            truth = w if w in model.vocab else UNK
            p = model.word_prob(s[:i], truth)
            n += 1
            if p > 0:
                log_sum += math.log(p)
            else:
                zeros += 1
            guess = [x for x, _ in model.top_k(s[:i], 5)]
            hit3 += truth in guess[:3]
            hit5 += truth in guess
    ppl = "∞" if zeros else f"{math.exp(-log_sum / n):.2f}"
    return {"model": name, "k": k, "perplexity": ppl,
            "top-3 (%)": round(100 * hit3 / n, 1), "top-5 (%)": round(100 * hit5 / n, 1)}


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.header("Settings")
corpus = st.sidebar.radio("Corpus", ["large", "small"], horizontal=True,
                          format_func=lambda c: {"large": "Large", "small": "Small (lab)"}[c])
n_sugg = st.sidebar.slider("Suggestions", 3, 10, DEFAULT_TOP_K)

with st.sidebar.expander("Advanced settings"):
    model_name = st.selectbox("Model", MODEL_NAMES)
    k = st.select_slider("Smoothing k", options=[0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0], value=0.1)
    min_count = MIN_CONTEXT_COUNT
    if model_name == BACKOFF:
        min_count = st.slider("Back-off min_count", 1, 50, MIN_CONTEXT_COUNT,
                              help="A context must be seen this many times to be trusted "
                                   "(1 = back off only if never seen).")

try:
    train, test = get_split(corpus)
except (FileNotFoundError, ValueError) as e:
    st.error(f"**Could not load the corpus.**\n\n```\n{e}\n```")
    st.stop()

model = KeyboardModel(corpus, model_name, k, min_count)
st.sidebar.caption(f"{len(train):,} training sentences · {len(model.vocab) - 2} words")

st.html('<div class="kb-eyebrow">Lab 02 · N-gram language model</div>'
        '<div class="kb-title">Smart Keyboard</div>'
        '<div class="kb-sub">Next-word suggestions from a trigram model with add-k smoothing and back-off.</div>')

tab_kb, tab_eval = st.tabs(["Keyboard", "Evaluation"])

# ----------------------------------------------------------------------------
# Keyboard
# ----------------------------------------------------------------------------
with tab_kb:
    st.session_state.setdefault("typed", "the students eat ")

    def insert(word, replace_last):
        parts = tokenize(st.session_state.typed)
        if word == EOS:                       # end of sentence -> start a new one
            st.session_state.typed = ""
            return
        if replace_last and parts:
            parts[-1] = word
        else:
            parts.append(word)
        st.session_state.typed = " ".join(parts) + " "

    def fix(index, word):
        parts = tokenize(st.session_state.typed)
        parts[index] = word
        st.session_state.typed = " ".join(parts) + " "

    text = st.text_input("Type a sentence", key="typed")
    tokens = tokenize(text)

    # Still typing an unknown word (no space after it) -> complete or correct it.
    completing = bool(tokens) and not text.endswith(" ") and tokens[-1] not in model.vocab
    history = tokens[:-1] if completing else tokens

    if completing:
        prefix = tokens[-1]
        matches = [w for w in model.vocab if w.startswith(prefix) and w not in (EOS, UNK)]
        matches += [w for w in did_you_mean(prefix, model.vocab) if w not in matches]
        suggestions = sorted(((w, model.word_prob(history, w)) for w in matches),
                             key=lambda x: -x[1])[:n_sugg]
    else:
        suggestions = model.top_k(history, n_sugg)

    if suggestions:
        cols = st.columns(len(suggestions))
        for i, (col, (w, p)) in enumerate(zip(cols, suggestions)):
            col.button("End" if w == EOS else w, key=f"s_{i}_{w}", on_click=insert, args=(w, completing),
                       help="end of sentence · " * (w == EOS) + f"probability {p:.1%}", width="stretch",
                       type="primary" if i == 0 else "secondary")
    else:
        st.caption(f"No known word looks like “{tokens[-1]}”.")

    # Unknown words already typed -> offer a correction.
    for i, w in enumerate(history):
        if w not in model.vocab:
            fixes = did_you_mean(w, model.vocab)
            if fixes:
                c = st.columns([2] + [1] * len(fixes) + [3 - len(fixes) + 1])
                c[0].markdown(f'<p class="kb-unknown"><b>{w}</b> — did you mean</p>', unsafe_allow_html=True)
                for col, f in zip(c[1:], fixes):
                    col.button(f, key=f"fix_{i}_{f}", on_click=fix, args=(i, f), width="stretch")
            else:
                st.caption(f"“{w}” is unknown, so the model reads it as {UNK} and keeps going.")

    if st.button("Auto-complete"):
        words = list(history)
        for _ in range(25):
            w = model.top_k(words, 1)[0][0]
            if w == EOS:
                break
            words.append(w)
            if w in (".", "?"):
                break
        st.success(" ".join(words))

    with st.expander("Why these suggestions?"):
        used, ctx = model.explain(history)
        seen = context_count(model.tri, history)
        st.markdown(f"**Model used:** {used}  \n"
                    f"**Context:** `{' '.join(ctx) if ctx else '(none)'}`  \n"
                    f"**Trigram context seen in training:** {seen} times")
        top10 = model.top_k(history, 10)
        fig, ax = plt.subplots(figsize=(6, 2.8))
        ax.barh(["End" if w == EOS else w for w, _ in top10], [p for _, p in top10],
                color=["#ff4b4b"] * n_sugg + ["#cccccc"] * (10 - n_sugg))
        ax.invert_yaxis(); ax.set_xlabel("P(word | context)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
with tab_eval:
    st.subheader(f"Held-out test documents · {corpus} corpus")
    if st.button("Run comparison"):
        configs = [("Unigram", 1.0), ("Bigram", 1.0), ("Trigram", 1.0), ("Trigram", 0.1),
                   ("Trigram", 0.0), (INTERP, 0.1), (BACKOFF, 0.1)]
        st.dataframe(pd.DataFrame([evaluate(corpus, n, kk, MIN_CONTEXT_COUNT) for n, kk in configs]),
                     hide_index=True, width="stretch")
        st.caption("Perplexity: lower is better. Accuracy: is the true next word among the suggestions?")

    chart = RESULTS / corpus / "ngram_results.png"
    if chart.exists():
        st.image(str(chart))
    else:
        st.caption(f"Run `python -m src.evaluate --corpus {corpus}` to add the results chart here.")
