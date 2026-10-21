"""Build the instructor corpus used by Labs 02, 03, 07 and 09.

Run from the lab folder:
    python data/make_data.py                 -> data/corpus.csv        (ORIGINAL small corpus, unchanged)
    python data/make_data.py --size large    -> data/corpus_large.csv  (NEW bigger corpus)
    python data/make_data.py --size both     -> both files

Output columns: doc_id, sentence, tags

The corpus is SYNTHETIC: sentences are sampled from a hand-written grammar (fixed seed 42),
so there are no licensing or privacy issues and every lab runs offline.
Consequences you must mention in your README:
  * the language is far more regular than real English -> perplexities are LOW
  * the vocabulary has semantic groups (animals, foods, people, places, objects) used in Lab 07
  * every word has a gold part-of-speech tag (used in Lab 03); "watch" is ambiguous (NOUN/VERB)
Sentences are grouped into "documents" of 20 sentences: train/test splits are made BY DOCUMENT.

WHY TWO GRAMMARS?
  The small grammar (V1) is shared with other labs, so it is kept byte-for-byte identical.
  The large grammar (V2) adds: ~3x more words, new word groups (drinks, meals, times),
  new sentence structures (questions, negation, "and"/"but", going somewhere),
  and THEMED documents (each document favours a topic) so the split-by-document lesson is stronger.
"""
import argparse
import csv
import os
import random
from collections import Counter

SEED = 42
N_SENTENCES = 1200
DOC_SIZE = 20

# =====================================================================================
# V1 GRAMMAR — ORIGINAL (do not modify: other labs depend on the exact same corpus.csv)
# =====================================================================================
ADJ = ["small", "hungry", "happy", "old", "quiet", "brave", "young", "lazy"]
ANIMALS = ["cats", "dogs", "foxes", "horses", "birds", "mice"]
FOODS = ["bread", "cheese", "apples", "soup", "rice", "cakes"]
PEOPLE = ["girls", "boys", "farmers", "teachers", "doctors", "children"]
PLACES = ["garden", "market", "school", "river", "kitchen", "park"]
OBJECTS = ["watch", "ball", "book", "lamp", "key", "coin"]
PREPS = ["in", "near", "behind", "at", "by"]
V_ANIMAL_INTR = ["sleep", "run", "play"]
V_ANIMAL_TR = ["chase", "follow", "fear"]
V_FOOD = ["eat", "cook", "buy", "sell", "bring"]
V_PLACE = ["visit", "clean", "leave"]
V_FIND = ["find", "buy", "bring", "lose"]

CATEGORIES = {"animal": ANIMALS, "food": FOODS, "person": PEOPLE, "place": PLACES, "object": OBJECTS}


def np_(rng, nouns, adj_prob=0.35):
    """DET (ADJ) NOUN  -> (words, tags)"""
    words, tags = ["the"], ["DET"]
    if rng.random() < adj_prob:
        words.append(rng.choice(ADJ)); tags.append("ADJ")
    words.append(rng.choice(nouns)); tags.append("NOUN")
    return words, tags


def pp(rng):
    p = rng.choice(PREPS)
    w, t = np_(rng, PLACES, 0.15)
    return [p] + w, ["PREP"] + t


def sentence(rng):
    kind = rng.choice(["a_intr", "a_tr", "p_food", "p_place", "p_watch", "a_eat", "p_find"])
    words, tags = [], []

    def add(w, t):
        words.extend(w); tags.extend(t)

    if kind == "a_intr":      # the happy cats sleep in the garden .
        add(*np_(rng, ANIMALS)); add([rng.choice(V_ANIMAL_INTR)], ["VERB"]); add(*pp(rng))
    elif kind == "a_tr":      # the dogs chase the cats .
        add(*np_(rng, ANIMALS)); add([rng.choice(V_ANIMAL_TR)], ["VERB"]); add(*np_(rng, ANIMALS, 0.2))
    elif kind == "p_food":    # the farmers sell the cheese at the market .
        add(*np_(rng, PEOPLE)); add([rng.choice(V_FOOD)], ["VERB"]); add(*np_(rng, FOODS, 0.2)); add(*pp(rng))
    elif kind == "p_place":   # the teachers visit the school .
        add(*np_(rng, PEOPLE)); add([rng.choice(V_PLACE)], ["VERB"]); add(*np_(rng, PLACES, 0.2))
    elif kind == "p_watch":   # the girls watch the birds in the park .   (watch = VERB)
        add(*np_(rng, PEOPLE)); add(["watch"], ["VERB"]); add(*np_(rng, ANIMALS, 0.2)); add(*pp(rng))
    elif kind == "a_eat":     # the mice eat the cheese in the kitchen .
        add(*np_(rng, ANIMALS)); add(["eat"], ["VERB"]); add(*np_(rng, FOODS, 0.2)); add(*pp(rng))
    else:                     # the children find the watch in the garden .  (watch = NOUN)
        add(*np_(rng, PEOPLE)); add([rng.choice(V_FIND)], ["VERB"]); add(*np_(rng, OBJECTS, 0.2)); add(*pp(rng))
    words.append("."); tags.append("PUNCT")
    return words, tags


# =====================================================================================
# V2 GRAMMAR — LARGE (new). Every V1 word is kept, so V1 sentences are still possible.
# =====================================================================================
ADJ_L = ADJ + ["big", "tired", "clever", "busy", "kind", "angry", "sleepy", "little",
               "noisy", "shy", "proud", "strong"]
ANIMALS_L = ANIMALS + ["rabbits", "ducks", "cows", "sheep", "goats", "chickens", "bears", "wolves"]
FOODS_L = FOODS + ["eggs", "fish", "meat", "pasta", "salad", "bananas", "oranges", "cookies", "pizza"]
MEALS = ["breakfast", "lunch", "dinner"]                      # used WITHOUT "the": "eat breakfast"
DRINKS = ["coffee", "tea", "milk", "water", "juice", "lemonade"]
PEOPLE_L = PEOPLE + ["students", "workers", "friends", "parents", "nurses", "cooks", "drivers", "artists"]
PLACES_L = PLACES + ["library", "beach", "forest", "city", "village", "station", "hospital",
                     "cafe", "farm", "museum"]
OBJECTS_L = OBJECTS + ["phone", "bag", "pen", "map", "hat", "umbrella", "bottle", "ticket",
                       "letter", "camera"]
PREPS_L = PREPS + ["inside", "outside", "under", "around"]

V_ANIMAL_INTR_L = V_ANIMAL_INTR + ["jump", "swim", "hide", "rest"]
V_ANIMAL_TR_L = V_ANIMAL_TR + ["bite", "help", "ignore", "love"]
V_FOOD_L = V_FOOD + ["prepare", "share", "order", "like", "want"]
V_MEAL = ["eat", "have", "cook", "skip", "prepare"]
V_DRINK = ["drink", "make", "order", "want", "like", "bring"]
V_PLACE_L = V_PLACE + ["enter", "paint", "explore", "love"]
V_FIND_L = V_FIND + ["carry", "need", "forget", "keep", "show"]
V_PERSON = ["meet", "help", "call", "thank", "visit"]
V_GO = ["go", "walk", "drive", "travel", "return"]
V_LIKE = ["like", "want", "need", "love"]
CONJ = ["and", "but", "while"]

# Time expressions are multi-word, so each one carries its own tags.
TIMES = [(["today"], ["ADV"]), (["yesterday"], ["ADV"]), (["tomorrow"], ["ADV"]),
         (["every", "morning"], ["DET", "NOUN"]), (["every", "evening"], ["DET", "NOUN"]),
         (["at", "night"], ["PREP", "NOUN"]), (["in", "the", "morning"], ["PREP", "DET", "NOUN"]),
         (["in", "the", "afternoon"], ["PREP", "DET", "NOUN"]), (["on", "sunday"], ["PREP", "NOUN"])]

CATEGORIES_L = {"animal": ANIMALS_L, "food": FOODS_L + MEALS, "drink": DRINKS, "person": PEOPLE_L,
                "place": PLACES_L, "object": OBJECTS_L}

# Each document gets a THEME: 60 % of its sentences come from the theme's sentence types.
# -> documents differ in vocabulary, which is exactly why we must split BY DOCUMENT.
THEMES = {
    "animals": ["a_intr", "a_tr", "a_eat", "p_watch", "conj"],
    "food": ["p_food", "p_meal", "p_drink", "a_eat", "neg"],
    "city": ["p_place", "p_go", "p_meet", "question"],
    "objects": ["p_find", "p_watch", "question", "neg"],
}
ALL_KINDS = sorted({k for kinds in THEMES.values() for k in kinds})
THEME_PROB = 0.6
TIME_PROB = 0.25


def np_l(rng, nouns, adj_prob=0.35):
    """DET (ADJ) NOUN with the large adjective list."""
    words, tags = ["the"], ["DET"]
    if rng.random() < adj_prob:
        words.append(rng.choice(ADJ_L)); tags.append("ADJ")
    words.append(rng.choice(nouns)); tags.append("NOUN")
    return words, tags


def pp_l(rng):
    """PREP DET (ADJ) PLACE"""
    w, t = np_l(rng, PLACES_L, 0.15)
    return [rng.choice(PREPS_L)] + w, ["PREP"] + t


def any_object(rng):
    """An object of 'like / want / need / love': any category (meals without 'the')."""
    group = rng.choice(["animal", "food", "meal", "drink", "place", "object"])
    if group == "meal":
        return [rng.choice(MEALS)], ["NOUN"]
    nouns = {"animal": ANIMALS_L, "food": FOODS_L, "drink": DRINKS,
             "place": PLACES_L, "object": OBJECTS_L}[group]
    return np_l(rng, nouns, 0.2)


def clause_l(rng, kind):
    """One declarative clause (no final punctuation) -> (words, tags)."""
    W, T = [], []

    def add(w, t):
        W.extend(w); T.extend(t)

    if kind == "a_intr":        # the brave rabbits hide under the bridge
        add(*np_l(rng, ANIMALS_L)); add([rng.choice(V_ANIMAL_INTR_L)], ["VERB"])
        if rng.random() < 0.7:
            add(*pp_l(rng))
    elif kind == "a_tr":        # the wolves chase the sheep
        add(*np_l(rng, ANIMALS_L)); add([rng.choice(V_ANIMAL_TR_L)], ["VERB"]); add(*np_l(rng, ANIMALS_L, 0.2))
    elif kind == "a_eat":       # the goats eat the salad in the farm
        add(*np_l(rng, ANIMALS_L)); add(["eat"], ["VERB"]); add(*np_l(rng, FOODS_L, 0.2)); add(*pp_l(rng))
    elif kind == "p_food":      # the cooks prepare the pasta in the kitchen
        add(*np_l(rng, PEOPLE_L)); add([rng.choice(V_FOOD_L)], ["VERB"]); add(*np_l(rng, FOODS_L, 0.2))
        add(*pp_l(rng))
    elif kind == "p_meal":      # the students eat breakfast in the cafe
        add(*np_l(rng, PEOPLE_L)); add([rng.choice(V_MEAL)], ["VERB"]); add([rng.choice(MEALS)], ["NOUN"])
        if rng.random() < 0.5:
            add(*pp_l(rng))
    elif kind == "p_drink":     # the friends drink the coffee at the cafe
        add(*np_l(rng, PEOPLE_L)); add([rng.choice(V_DRINK)], ["VERB"]); add(*np_l(rng, DRINKS, 0.1))
        if rng.random() < 0.5:
            add(*pp_l(rng))
    elif kind == "p_place":     # the artists paint the museum
        add(*np_l(rng, PEOPLE_L)); add([rng.choice(V_PLACE_L)], ["VERB"]); add(*np_l(rng, PLACES_L, 0.2))
    elif kind == "p_go":        # the workers go to the station
        add(*np_l(rng, PEOPLE_L)); add([rng.choice(V_GO)], ["VERB"]); add(["to"], ["PREP"])
        add(*np_l(rng, PLACES_L, 0.15))
    elif kind == "p_meet":      # the nurses meet the doctors at the hospital
        add(*np_l(rng, PEOPLE_L)); add([rng.choice(V_PERSON)], ["VERB"]); add(*np_l(rng, PEOPLE_L, 0.2))
        if rng.random() < 0.5:
            add(*pp_l(rng))
    elif kind == "p_watch":     # the girls watch the ducks in the park   (watch = VERB)
        add(*np_l(rng, PEOPLE_L)); add(["watch"], ["VERB"]); add(*np_l(rng, ANIMALS_L, 0.2)); add(*pp_l(rng))
    elif kind == "p_find":      # the drivers lose the ticket at the station   (watch = NOUN possible)
        add(*np_l(rng, PEOPLE_L)); add([rng.choice(V_FIND_L)], ["VERB"]); add(*np_l(rng, OBJECTS_L, 0.2))
        add(*pp_l(rng))
    else:
        raise ValueError(f"unknown clause kind: {kind}")
    return W, T


def sentence_l(rng, theme):
    """One sentence of the large grammar, biased towards the document's theme."""
    kinds = THEMES[theme] if rng.random() < THEME_PROB else ALL_KINDS
    kind = rng.choice(kinds)

    if kind == "question":      # do the students like the pizza ?
        W, T = ["do"], ["AUX"]
        w, t = np_l(rng, PEOPLE_L); W += w; T += t
        W.append(rng.choice(V_LIKE)); T.append("VERB")
        w, t = any_object(rng); W += w; T += t
        return W + ["?"], T + ["PUNCT"]

    if kind == "neg":           # the boys do not like the soup .
        W, T = np_l(rng, PEOPLE_L)
        W += ["do", "not", rng.choice(V_LIKE)]; T += ["AUX", "PART", "VERB"]
        w, t = any_object(rng); W += w; T += t
    elif kind == "conj":        # the cats sleep and the dogs play .
        W, T = np_l(rng, ANIMALS_L)
        W.append(rng.choice(V_ANIMAL_INTR_L)); T.append("VERB")
        W.append(rng.choice(CONJ)); T.append("CONJ")
        w, t = np_l(rng, ANIMALS_L); W += w; T += t
        W.append(rng.choice(V_ANIMAL_INTR_L)); T.append("VERB")
    else:
        W, T = clause_l(rng, kind)

    if rng.random() < TIME_PROB:  # optional time expression: "... every morning ."
        w, t = rng.choice(TIMES); W += w; T += t
    return W + ["."], T + ["PUNCT"]


# =====================================================================================
# Writing + statistics
# =====================================================================================
def write_small(path="data/corpus.csv"):
    """Original corpus — same seed, same order, same output as before."""
    rng = random.Random(SEED)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "sentence", "tags"])
        for i in range(N_SENTENCES):
            words, tags = sentence(rng)
            w.writerow([i // DOC_SIZE, " ".join(words), " ".join(tags)])
    return path


def write_large(path="data/corpus_large.csv", n_sentences=6000):
    """Large corpus: themed documents of DOC_SIZE sentences."""
    rng = random.Random(SEED)
    n_docs = n_sentences // DOC_SIZE
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "sentence", "tags"])
        for doc in range(n_docs):
            theme = rng.choice(sorted(THEMES))
            for _ in range(DOC_SIZE):
                words, tags = sentence_l(rng, theme)
                assert len(words) == len(tags), (words, tags)   # every word must have a tag
                w.writerow([doc, " ".join(words), " ".join(tags)])
    return path


def print_stats(path):
    tokens, docs, n_sent = Counter(), set(), 0
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            docs.add(row["doc_id"]); n_sent += 1
            tokens.update(row["sentence"].split())
    print(f"wrote {path}: {n_sent} sentences in {len(docs)} documents, "
          f"{sum(tokens.values())} tokens, {len(tokens)} word types")


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic corpus.")
    ap.add_argument("--size", choices=["small", "large", "both"], default="small",
                    help="small = original corpus.csv, large = corpus_large.csv")
    ap.add_argument("--sentences", type=int, default=6000, help="number of sentences for the large corpus")
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    if args.size in ("small", "both"):
        print_stats(write_small())
    if args.size in ("large", "both"):
        print_stats(write_large(n_sentences=args.sentences))


if __name__ == "__main__":
    main()
