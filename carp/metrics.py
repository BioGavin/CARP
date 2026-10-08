import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    ndcg_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from scipy.stats import pearsonr, kendalltau, spearmanr


def regression_metrics(pred, true):
    pred = np.asarray(pred, dtype=float).ravel()
    true = np.asarray(true, dtype=float).ravel()
    mse = mean_squared_error(true, pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(true, pred)),
        "pcc": float(pearsonr(pred, true)[0]),
        "ktc": float(kendalltau(pred, true)[0]),
        "spearman": float(spearmanr(pred, true)[0]),
        "n": int(len(true)),
    }


def ranking_metrics(pred, true, ks=(10, 50, 100)):
    pred = np.asarray(pred, dtype=float).ravel()
    true = np.asarray(true, dtype=float).ravel()
    n = len(true)
    out = {}
    order_pred = np.argsort(-pred)
    order_true = np.argsort(-true)
    rel = (true - true.min())[None, :]
    for k in ks:
        if k > n:
            continue
        top_pred = set(order_pred[:k].tolist())
        top_true = set(order_true[:k].tolist())
        out[f"top{k}_overlap"] = len(top_pred & top_true) / k
        out[f"ndcg{k}"] = float(ndcg_score(rel, pred[None, :], k=k))
    return out


def precision_at_k(prob, label, ks=(10, 50, 100)):
    prob = np.asarray(prob, dtype=float).ravel()
    label = np.asarray(label, dtype=int).ravel()
    n = len(label)
    lab_sorted = label[np.argsort(-prob, kind="stable")]  # highest score first
    out = {"prevalence": float(label.mean()) if n else float("nan")}
    for k in ks:
        if k <= n:
            out[f"p@{k}"] = float(lab_sorted[:k].mean())
    return out


def classification_metrics(prob, label, threshold=0.5, ks=(50, 100, 200, 400)):
    prob = np.asarray(prob, dtype=float).ravel()
    label = np.asarray(label, dtype=float).ravel()
    pred = (prob >= threshold).astype(int)
    m = {
        "auc": float(roc_auc_score(label, prob)),
        "aupr": float(average_precision_score(label, prob)),
        "f1": float(f1_score(label, pred, zero_division=0)),
        "precision": float(precision_score(label, pred, zero_division=0)),
        "recall": float(recall_score(label, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(label, pred)),
        "acc": float((pred == label).mean()),
        "n": int(len(label)),
    }
    m.update(precision_at_k(prob, label, ks))  # adds p@k + prevalence
    return m
