"""
main.py  -  AI-Powered Fake News Detection Using Text Classification
====================================================================
End-to-end pipeline for Project 1 of the IICT Summer Internship 2026.

Everything the brief calls for is implemented FROM SCRATCH in `mlcore/`:
manual tokenisation, Bag-of-Words, TF-IDF, LSA embeddings, KNN, Logistic
Regression, Random Forest and a one-hidden-layer Neural Network, together
with the metric suite.  scikit-learn is used only at the very end as an
independent cross-check that the from-scratch implementations are correct.

Run
---
    python generate_sample_dataset.py     # once, if data/ is empty
    python main.py

Outputs
-------
    results/*.png        figures for the report and the slide deck
    results/metrics.csv  full results table
    results/metrics.json machine-readable results
    results/summary.txt  console log

Author : IICT Summer Internship 2026 (AI & ML)
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlcore.preprocessing import preprocess, train_test_split, STOPWORDS
from mlcore.vectorizers import CountVectorizer, TfidfVectorizer, LSAEmbedding
from mlcore.models import (KNNClassifier, LogisticRegression,
                           RandomForestClassifier, NeuralNetwork)
from mlcore import metrics as M

# --------------------------------------------------------------------------- #
CONFIG = {
    "max_features":   3000,
    "min_df":         3,
    "ngram_range":    (1, 2),
    "test_size":      0.20,
    "random_state":   42,
    "lsa_components": 150,
    "knn_k":          7,
    "rf_trees":       30,
    "rf_depth":       10,
    "nn_hidden":      96,
    "nn_epochs":      35,
    "cv_folds":       5,
}
LABELS = ["REAL (0)", "FAKE (1)"]
RESULTS = "results"


def log(msg: str, buf: list[str]):
    print(msg)
    buf.append(msg)


# --------------------------------------------------------------------------- #
# WEEK 1  -  Data loading and cleaning
# --------------------------------------------------------------------------- #
def load_dataset(buf) -> pd.DataFrame:
    """Prefer the real Kaggle file; fall back to the reproducible sample."""
    for path, note in (("data/train.csv", "Kaggle 'Fake and Real News' dataset"),
                       ("data/news_dataset.csv", "synthetic sample corpus")):
        if os.path.exists(path):
            df = pd.read_csv(path)
            log(f"[Week 1] Loaded {path}  ({note})", buf)
            break
    else:
        raise FileNotFoundError("No dataset found. Run generate_sample_dataset.py first.")

    # Kaggle's file uses the same column names; guard against missing ones.
    if "text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column.")
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column (0 = real, 1 = fake).")

    # Public corpora ship the label as the strings REAL / FAKE; map to 0 / 1.
    if df["label"].dtype == object:
        df["label"] = (df["label"].astype(str).str.strip().str.upper()
                       .map({"FAKE": 1, "REAL": 0, "1": 1, "0": 0}))
        df = df.dropna(subset=["label"])

    before = len(df)
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
    df["label"] = df["label"].astype(int)
    log(f"[Week 1] Dropped {before - len(df)} null/duplicate rows -> {len(df)} articles", buf)
    return df.reset_index(drop=True)


def exploratory_analysis(df: pd.DataFrame, tokens: list[list[str]], buf):
    """Week-2 EDA: class balance, document length, class-conditional vocabulary."""
    counts = {"REAL": int((df.label == 0).sum()), "FAKE": int((df.label == 1).sum())}
    log(f"[Week 2] Class balance          : {counts}", buf)

    lengths = np.array([len(t) for t in tokens])
    log(f"[Week 2] Tokens per article     : mean {lengths.mean():.1f}, "
        f"median {np.median(lengths):.0f}, max {lengths.max()}", buf)

    real_vocab = Counter(t for toks, y in zip(tokens, df.label) if y == 0 for t in toks)
    fake_vocab = Counter(t for toks, y in zip(tokens, df.label) if y == 1 for t in toks)
    log(f"[Week 2] Vocabulary size        : REAL {len(real_vocab)}, FAKE {len(fake_vocab)}", buf)
    log(f"[Week 2] Top REAL terms         : {[w for w, _ in real_vocab.most_common(8)]}", buf)
    log(f"[Week 2] Top FAKE terms         : {[w for w, _ in fake_vocab.most_common(8)]}", buf)

    M.plot_class_balance(counts, f"{RESULTS}/fig1_class_balance.png",
                         "Corpus class balance")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.hist([lengths[df.label == 0], lengths[df.label == 1]], bins=28,
            label=["REAL", "FAKE"], color=["#1F3A5F", "#E4572E"])
    ax.set_xlabel("Tokens per article (after stop-word removal)")
    ax.set_ylabel("Frequency")
    ax.set_title("Document length distribution", fontsize=11, weight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/fig2_length_distribution.png", dpi=170)
    plt.close(fig)
    return counts


# --------------------------------------------------------------------------- #
# WEEK 3 / 4  -  Training and evaluation
# --------------------------------------------------------------------------- #
def evaluate(name, model, X_tr, y_tr, X_te, y_te, buf, results, curves):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    fit_s = time.time() - t0

    t0 = time.time()
    pred = model.predict(X_te)
    pred_s = time.time() - t0
    proba = model.predict_proba(X_te)[:, 1]

    cm = M.confusion_matrix(y_te, pred)
    fpr, tpr = M.roc_curve(y_te, proba)
    auc = M.roc_auc(y_te, proba)

    results[name] = {
        "accuracy":  M.accuracy(y_te, pred),
        "precision": M.precision(y_te, pred),
        "recall":    M.recall(y_te, pred),
        "f1":        M.f1(y_te, pred),
        "specificity": M.specificity(y_te, pred),
        "roc_auc":   auc,
        "fit_seconds":     round(fit_s, 3),
        "predict_seconds": round(pred_s, 3),
        "confusion_matrix": cm.tolist(),
    }
    curves[name] = (fpr, tpr, auc)

    r = results[name]
    log(f"\n--- {name} ---", buf)
    log(f"  accuracy {r['accuracy']:.4f} | precision {r['precision']:.4f} | "
        f"recall {r['recall']:.4f} | F1 {r['f1']:.4f} | AUC {auc:.4f}", buf)
    log(f"  fit {fit_s:.2f}s | predict {pred_s:.3f}s", buf)
    log(M.classification_report(y_te, pred, LABELS), buf)

    M.plot_confusion_matrix(
        cm, ["REAL", "FAKE"], f"Confusion matrix - {name}",
        f"{RESULTS}/cm_{name.lower().replace(' ', '_')}.png")
    return model


def cross_validate(X, y, buf):
    """Manual stratified 5-fold CV on the strongest linear model."""
    scores = []
    for tr, va in M.k_fold_indices(y, k=CONFIG["cv_folds"],
                                   random_state=CONFIG["random_state"]):
        m = LogisticRegression(lr=3.0, n_iters=2000).fit(X[tr], y[tr])
        scores.append(M.f1(y[va], m.predict(X[va])))
    scores = np.array(scores)
    log(f"\n[Week 4] {CONFIG['cv_folds']}-fold CV F1 (Logistic Regression): "
        f"{scores.mean():.4f} +/- {scores.std():.4f}   folds={np.round(scores, 4).tolist()}", buf)
    return scores


def sklearn_crosscheck(X_tr, y_tr, X_te, y_te, buf):
    """Independent verification that the from-scratch models are not buggy."""
    try:
        from sklearn.linear_model import LogisticRegression as SkLR
        from sklearn.ensemble import RandomForestClassifier as SkRF
        from sklearn.neighbors import KNeighborsClassifier as SkKNN
        from sklearn.neural_network import MLPClassifier as SkMLP
    except ImportError:
        log("\n[Cross-check] scikit-learn unavailable - skipped.", buf)
        return {}

    ref = {
        "KNN":            SkKNN(n_neighbors=CONFIG["knn_k"], metric="cosine"),
        "LogReg":         SkLR(max_iter=1000),
        "Random Forest":  SkRF(n_estimators=CONFIG["rf_trees"],
                               max_depth=CONFIG["rf_depth"], random_state=42),
        "Neural Net":     SkMLP(hidden_layer_sizes=(CONFIG["nn_hidden"],),
                                max_iter=300, random_state=42),
    }
    out = {}
    log("\n[Cross-check] scikit-learn reference accuracies", buf)
    for name, m in ref.items():
        m.fit(X_tr, y_tr)
        acc = M.accuracy(y_te, m.predict(X_te))
        out[name] = acc
        log(f"  {name:<15} {acc:.4f}", buf)
    return out


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(RESULTS, exist_ok=True)
    buf: list[str] = []
    log("=" * 74, buf)
    log("AI-POWERED FAKE NEWS DETECTION USING TEXT CLASSIFICATION", buf)
    log("IICT Summer Internship 2026  |  Project 1", buf)
    log("=" * 74, buf)

    # ---------------- Week 1: load + clean ---------------- #
    df = load_dataset(buf)
    t0 = time.time()
    tokens = [preprocess(t, stem=True, drop_stopwords=True) for t in df["text"]]
    log(f"[Week 1] Manual preprocessing of {len(tokens)} documents "
        f"in {time.time() - t0:.1f}s (stop-list size {len(STOPWORDS)})", buf)

    # ---------------- Week 2: EDA + features -------------- #
    exploratory_analysis(df, tokens, buf)

    y = df["label"].to_numpy()
    idx = np.arange(len(y))
    idx_tr, idx_te, y_tr, y_te = train_test_split(
        idx, y, test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"], stratify=True)
    docs_tr = [tokens[i] for i in idx_tr]
    docs_te = [tokens[i] for i in idx_te]
    log(f"\n[Week 2] Split: {len(idx_tr)} train / {len(idx_te)} test "
        f"(stratified, test_size={CONFIG['test_size']})", buf)

    bow = CountVectorizer(max_features=CONFIG["max_features"],
                          min_df=CONFIG["min_df"], ngram_range=CONFIG["ngram_range"])
    Xb_tr = bow.fit_transform(docs_tr)
    log(f"[Week 2] Bag-of-Words matrix    : {Xb_tr.shape}, "
        f"sparsity {100 * (Xb_tr == 0).mean():.1f}%", buf)

    tfidf = TfidfVectorizer(max_features=CONFIG["max_features"],
                            min_df=CONFIG["min_df"], ngram_range=CONFIG["ngram_range"])
    X_tr = tfidf.fit_transform(docs_tr)          # fit on TRAIN only - no leakage
    X_te = tfidf.transform(docs_te)
    names = tfidf.get_feature_names()
    log(f"[Week 2] TF-IDF matrix          : {X_tr.shape}, "
        f"sparsity {100 * (X_tr == 0).mean():.1f}%", buf)

    lsa = LSAEmbedding(n_components=CONFIG["lsa_components"])
    E_tr = lsa.fit_transform(X_tr)
    E_te = lsa.transform(X_te)
    log(f"[Week 2] LSA embeddings         : {E_tr.shape}, "
        f"variance retained {lsa.explained_variance_ratio_.sum():.1%}", buf)

    # ---------------- Week 3: models ---------------------- #
    log("\n" + "=" * 74, buf)
    log("[Week 3] TRAINING FOUR FROM-SCRATCH CLASSIFIERS", buf)
    log("=" * 74, buf)

    results, curves = {}, {}
    evaluate("KNN", KNNClassifier(n_neighbors=CONFIG["knn_k"]),
             X_tr, y_tr, X_te, y_te, buf, results, curves)
    logreg = evaluate("Logistic Regression",
                      LogisticRegression(lr=3.0, n_iters=2500, verbose=False),
                      X_tr, y_tr, X_te, y_te, buf, results, curves)
    rf = evaluate("Random Forest",
                  RandomForestClassifier(n_estimators=CONFIG["rf_trees"],
                                         max_depth=CONFIG["rf_depth"],
                                         random_state=42),
                  X_tr, y_tr, X_te, y_te, buf, results, curves)
    # The neural net is trained on the dense LSA embeddings: a fully connected
    # net on the 1200-d sparse TF-IDF matrix overfits badly and trains far
    # more slowly for no gain.
    nn = evaluate("Neural Network",
                  NeuralNetwork(hidden=CONFIG["nn_hidden"],
                                epochs=CONFIG["nn_epochs"], lr=3e-3),
                  E_tr, y_tr, E_te, y_te, buf, results, curves)

    # ---------------- Week 4: analysis -------------------- #
    log("\n" + "=" * 74, buf)
    log("[Week 4] EVALUATION, VISUALISATION AND COMPARISON", buf)
    log("=" * 74, buf)

    M.plot_model_comparison(results, f"{RESULTS}/fig3_model_comparison.png",
                            "Model comparison on the held-out test set")
    M.plot_roc_curves(curves, f"{RESULTS}/fig4_roc_curves.png",
                      "ROC curves - fake news detection")
    M.plot_learning_curve({"Logistic Regression": logreg.loss_history_},
                          f"{RESULTS}/fig5_logreg_loss.png",
                          "Logistic regression convergence", "Gradient-descent iteration")
    M.plot_learning_curve({"Neural Network": nn.loss_history_},
                          f"{RESULTS}/fig6_nn_loss.png",
                          "Neural network training loss", "Epoch")

    pos, neg = logreg.top_features(names, k=14)
    M.plot_top_features(pos, neg, f"{RESULTS}/fig7_top_features.png",
                        "Most influential terms (logistic regression)",
                        "Pushes toward FAKE", "Pushes toward REAL")
    log(f"\n[Week 4] Strongest FAKE indicators : {[w for w, _ in pos[:8]]}", buf)
    log(f"[Week 4] Strongest REAL indicators : {[w for w, _ in neg[:8]]}", buf)

    imp = rf.feature_importances_()
    top_rf = np.argsort(-imp)[:10]
    log(f"[Week 4] Random-forest top splits  : {[names[i] for i in top_rf]}", buf)

    cv = cross_validate(X_tr, y_tr, buf)
    sk = sklearn_crosscheck(X_tr, y_tr, X_te, y_te, buf)

    best = max(results, key=lambda k: results[k]["f1"])
    log(f"\n[Week 4] BEST MODEL BY F1 : {best} "
        f"(F1 = {results[best]['f1']:.4f}, accuracy = {results[best]['accuracy']:.4f})", buf)

    # ---------------- persist ----------------------------- #
    table = pd.DataFrame(results).T[
        ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc",
         "fit_seconds", "predict_seconds"]].round(4)
    table.to_csv(f"{RESULTS}/metrics.csv")
    with open(f"{RESULTS}/metrics.json", "w") as f:
        json.dump({"config": {k: list(v) if isinstance(v, tuple) else v
                              for k, v in CONFIG.items()},
                   "dataset": {"n": int(len(df)),
                               "real": int((df.label == 0).sum()),
                               "fake": int((df.label == 1).sum()),
                               "n_train": int(len(idx_tr)), "n_test": int(len(idx_te)),
                               "vocab": int(len(names))},
                   "results": results,
                   "cv_f1_mean": float(cv.mean()), "cv_f1_std": float(cv.std()),
                   "sklearn_reference": sk,
                   "best_model": best,
                   "top_fake_terms": [w for w, _ in pos[:12]],
                   "top_real_terms": [w for w, _ in neg[:12]],
                   "lsa_variance": float(lsa.explained_variance_ratio_.sum())},
                  f, indent=2)

    log("\n" + table.to_string(), buf)
    with open(f"{RESULTS}/summary.txt", "w") as f:
        f.write("\n".join(buf))
    print(f"\nSaved figures + metrics to ./{RESULTS}/")


if __name__ == "__main__":
    main()
