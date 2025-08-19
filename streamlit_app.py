#shift_based_prediction.py

#Run: streamlit run shift_based_prediction.py

import itertools as it
import math
from collections import Counter
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Shift-Based Prediction System", layout="wide")
st.title("⚙️ Shift-Based Prediction System (Pick 3 / Pick 4)")
st.caption("Learns digit-to-digit transitions (not position-based) for lottery prediction.")

# -----------------------------
# Utilities
# -----------------------------

def to_tuple(x: str, n_digits: int) -> Tuple[int, ...]:
    s = str(x).strip()
    digs = [int(ch) for ch in s if ch.isdigit()]
    if len(digs) < n_digits:
        digs = [0] * (n_digits - len(digs)) + digs
    return tuple(digs[-n_digits:])

def tuple_to_str(t: Tuple[int, ...]) -> str:
    return ''.join(str(d) for d in t)

def greedy_multiset_mapping(a: Tuple[int, ...], b: Tuple[int, ...]):
    ca, cb = Counter(a), Counter(b)
    pairs = []
    for d in range(10):
        m = min(ca[d], cb[d])
        for _ in range(m):
            pairs.append((d, d))
        ca[d] -= m
        cb[d] -= m
    for x in ca:
        for _ in range(ca[x]):
            for y in cb:
                if cb[y] > 0:
                    pairs.append((x, y))
                    cb[y] -= 1
                    break
    return pairs

def extract_digit_transitions(draws: List[Tuple[int, ...]], lag: int) -> Counter:
    trans = Counter()
    for i in range(len(draws) - lag):
        a, b = draws[i], draws[i+lag]
        pairs = greedy_multiset_mapping(a, b)
        for x, y in pairs:
            trans[(x, y)] += 1
    return trans

def normalize_matrix(cnt: Counter) -> Dict[Tuple[int, int], float]:
    totals = Counter()
    for (x, y), c in cnt.items():
        totals[x] += c
    probs = {}
    for (x, y), c in cnt.items():
        denom = totals[x] if totals[x] else 1
        probs[(x, y)] = c / denom
    return probs

def apply_positionless_transitions(seed: Tuple[int, ...], probs: Dict[Tuple[int,int], float], top_k: int = 3) -> List[Tuple[int, ...]]:
    choices_per_digit = []
    for v in seed:
        dist = [(y, probs.get((v, y), 0.0)) for y in range(10)]
        dist.sort(key=lambda t: t[1], reverse=True)
        choices_per_digit.append([y for y, _ in dist[:top_k]])
    combos = list(it.product(*choices_per_digit))
    return combos

def score_by_transition_likelihood(cand: Tuple[int, ...], seed: Tuple[int, ...], probs: Dict[Tuple[int,int], float]) -> float:
    pairs = greedy_multiset_mapping(seed, cand)
    score = 0.0
    for x, y in pairs:
        score += probs.get((x, y), 0.0)
    return score

# -----------------------------
# Sidebar Controls
# -----------------------------

st.sidebar.header("History Input")
mode = st.sidebar.selectbox("Game", ["Pick 3", "Pick 4"], index=1)
n_digits = 3 if mode == "Pick 3" else 4

pasted = st.sidebar.text_area("Paste draws here (top = latest, e.g. M:2416, E:9724)")
recent_window = st.sidebar.slider("Recent window for speed detection (draws)", 5, 100, 20, 1)
max_lag = st.sidebar.slider("Max lag for transitions", 1, 20, 5, 1)
per_digit_topk = st.sidebar.slider("Per-digit top K", 1, 5, 3, 1)
num_preds = st.sidebar.slider("Number of predictions per label", 1, 20, 5, 1)

# -----------------------------
# Parse Draws
# -----------------------------

draws_by_label = {}
if pasted:
    lines = [line.strip() for line in pasted.split("\n") if line.strip()]
    for line in lines:
        if ":" in line:
            label, val = line.split(":", 1)
            label = label.strip()
            digs = to_tuple(val, n_digits)
            if label not in draws_by_label:
                draws_by_label[label] = []
            draws_by_label[label].append(digs)

all_labels = list(draws_by_label.keys())
if all_labels:
    filter_choice = st.sidebar.selectbox("Analyze draws from:", ["All"] + all_labels)

    # Use selected draws
    if filter_choice == "All":
        selected_labels = all_labels
    else:
        selected_labels = [filter_choice]

    st.subheader("Transition Analysis")
    combined_draws = [d for lbl in selected_labels for d in draws_by_label[lbl]]

    all_trans = Counter()
    for lag in range(1, max_lag+1):
        trans = extract_digit_transitions(combined_draws[-recent_window:], lag)
        all_trans.update(trans)
    probs = normalize_matrix(all_trans)

    st.subheader("Predictions")

    if filter_choice == "All":
        for lbl in all_labels:
            if not draws_by_label[lbl]:
                continue
            last_draw = draws_by_label[lbl][-1]
            candidates = apply_positionless_transitions(last_draw, probs, per_digit_topk)
            scored = [(cand, score_by_transition_likelihood(cand, last_draw, probs)) for cand in candidates]
            scored.sort(key=lambda t: t[1], reverse=True)
            preds = [tuple_to_str(c) for c, _ in scored[:num_preds]]

            st.markdown(f"### Predictions for {lbl}")
            st.write(f"Last draw: {tuple_to_str(last_draw)}")
            st.write(preds)
            st.download_button(f"Download Predictions ({lbl})", pd.Series(preds).to_csv(index=False), f"predictions_{lbl}.csv")
    else:
        lbl = selected_labels[0]
        last_draw = draws_by_label[lbl][-1]
        candidates = apply_positionless_transitions(last_draw, probs, per_digit_topk)
        scored = [(cand, score_by_transition_likelihood(cand, last_draw, probs)) for cand in candidates]
        scored.sort(key=lambda t: t[1], reverse=True)
        preds = [tuple_to_str(c) for c, _ in scored[:num_preds]]

        st.markdown(f"### Predictions for {lbl}")
        st.write(f"Last draw: {tuple_to_str(last_draw)}")
        st.write(preds)
        st.download_button(f"Download Predictions ({lbl})", pd.Series(preds).to_csv(index=False), f"predictions_{lbl}.csv")
else:
    st.info("Paste your draw history to start analysis.")
