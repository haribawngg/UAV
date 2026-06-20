"""
Shared utilities for Feature Selection (02_*) and Feature Engineering (03_*) notebooks.

Usage in a notebook cell (see any 02_*/03_* notebook's first cell for the
full Colab-aware bootstrap that precedes this):
    sys.path.insert(0, CODE_DIR + '/notebook')   # ensure notebook/ dir is importable
    from common import *

Design goals
------------
1. Absolute paths -> works regardless of Jupyter's current working directory,
   and resolves to Google Drive automatically when running on Colab.
2. Incremental, resumable experiment runner -> a crash (or Colab disconnect)
   loses at most one (K, seed, classifier) combination, not the whole
   notebook's progress.
3. One shared implementation of resampling / metrics / model factory so every
   FS and FE notebook behaves identically (fair comparison).
"""
import warnings
warnings.filterwarnings('ignore')

import os, sys, subprocess, time, json, pickle
import numpy as np
import pandas as pd

# ── Colab environment setup (no-op when run locally) ───────────────────────────
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    # Colab doesn't ship these two by default; everything else used below
    # (xgboost, lightgbm, sklearn, scipy) is preinstalled.
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                     'imbalanced-learn', 'shap'], check=True)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, average_precision_score
)
from sklearn.preprocessing import label_binarize
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import BorderlineSMOTE

# ── Absolute paths (robust regardless of Jupyter's working directory) ──────────
# Colab: data/processed/models/results live on Drive so they persist across
# sessions/disconnects. Local: defaults to this project's checkout path,
# overridable with the UAV_BASE_DIR env var.
BASE_DIR = ('/content/drive/MyDrive/UAV_data/' if IN_COLAB
            else os.environ.get('UAV_BASE_DIR', '/home/hanh/UAV_/'))
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed/')
RESULTS_DIR   = os.path.join(BASE_DIR, 'results/')
MODELS_DIR    = os.path.join(BASE_DIR, 'models/')
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR,    exist_ok=True)

# ── Experiment configuration (shared across ALL notebooks) ─────────────────────
SEEDS               = [42, 52, 62, 72, 82]
K_VALUES             = [4, 8, 16]
CAP_MAJORITY         = 20_000   # RandomUnderSampler cap for majority classes
MIN_MINORITY         = 10_000   # BorderlineSMOTE floor for minority classes
N_JOBS               = -1
RANKING_SAMPLE_SIZE  = 100_000  # stratified subsample size used ONLY to compute
                                 # expensive FS rankings (SHAP, RFE); does NOT
                                 # affect the actual classifier training data

METRICS = ['accuracy', 'precision', 'recall', 'f1', 'pr_auc', 'fpr', 'fnr',
           'train_time_s', 'inference_ms']


# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    """Load preprocessed splits, label encoder, metadata, and tuned hyperparameters."""
    splits_path = f'{PROCESSED_DIR}splits.npz'
    if not os.path.exists(splits_path):
        raise FileNotFoundError(
            f'{splits_path} not found. Run 00_preprocessing.ipynb first.'
        )
    data = np.load(splits_path)
    out = dict(
        X_train=data['X_train'], X_val=data['X_val'], X_test=data['X_test'],
        y_train=data['y_train'], y_val=data['y_val'], y_test=data['y_test'],
    )
    with open(f'{PROCESSED_DIR}label_encoder.pkl', 'rb') as f:
        out['le'] = pickle.load(f)
    with open(f'{PROCESSED_DIR}meta.json') as f:
        meta = json.load(f)

    best_params_path = f'{MODELS_DIR}best_params.json'
    if not os.path.exists(best_params_path):
        raise FileNotFoundError(
            f'{best_params_path} not found. Run 01_baseline.ipynb first.'
        )
    with open(best_params_path) as f:
        best_params = json.load(f)

    out['meta']             = meta
    out['best_params']      = best_params
    out['class_names']      = meta['classes']
    out['feature_names']    = meta['feature_names']
    out['n_classes']        = meta['n_classes']
    out['classifier_names'] = list(best_params.keys())
    return out


# ─────────────────────────────────────────────────────────────────────────────
def stratified_subsample(X, y, n, seed=42):
    """Stratified subsample of (X, y), size <= n. Speeds up expensive FS rankings."""
    if len(X) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    frac = n / len(y)
    idx_parts = []
    for cls, cnt in zip(classes, counts):
        cls_idx = np.where(y == cls)[0]
        take = min(len(cls_idx), max(1, int(round(cnt * frac))))
        idx_parts.append(rng.choice(cls_idx, take, replace=False))
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    return X[idx], y[idx]


# ─────────────────────────────────────────────────────────────────────────────
def resample_train(X_tr, y_tr, seed, cap=CAP_MAJORITY, min_count=MIN_MINORITY):
    """RandomUnderSampler (cap majority) -> BorderlineSMOTE (lift minority). Train only."""
    counts = pd.Series(y_tr).value_counts()
    under_strategy = {cls: min(int(cnt), cap) for cls, cnt in counts.items()}
    rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=seed)
    X_u, y_u = rus.fit_resample(X_tr, y_tr)

    counts_u = pd.Series(y_u).value_counts()
    smote_strategy = {cls: min_count for cls, cnt in counts_u.items() if cnt < min_count}
    if smote_strategy:
        bsmote = BorderlineSMOTE(sampling_strategy=smote_strategy, random_state=seed, n_jobs=N_JOBS)
        X_r, y_r = bsmote.fit_resample(X_u, y_u)
    else:
        X_r, y_r = X_u, y_u
    return X_r, y_r


# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, n_classes, y_prob=None):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    pr_auc = np.nan
    if y_prob is not None:
        y_bin  = label_binarize(y_true, classes=np.arange(n_classes))
        pr_auc = average_precision_score(y_bin, y_prob, average='macro')
    cm  = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    fp  = cm.sum(axis=0) - np.diag(cm)
    fn  = cm.sum(axis=1) - np.diag(cm)
    tp  = np.diag(cm)
    tn  = cm.sum() - (fp + fn + tp)
    fpr = np.nanmean(fp / (fp + tn + 1e-12))
    fnr = np.nanmean(fn / (fn + tp + 1e-12))
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1,
                pr_auc=pr_auc, fpr=fpr, fnr=fnr)


def timed_predict(model, X):
    _ = model.predict(X[:10])  # warm-up (excluded from timing)
    t0 = time.perf_counter()
    y_pred = model.predict(X)
    inf_ms = (time.perf_counter() - t0) / len(X) * 1000
    y_prob = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
    return y_pred, y_prob, inf_ms


def make_model(clf_name, seed, best_params):
    params = {**best_params[clf_name]}
    if clf_name in ('RF', 'XGB', 'LGBM', 'KNN'):
        params['n_jobs'] = N_JOBS
    # Remove keys passed explicitly below to avoid duplicate keyword arguments
    params.pop('verbosity', None)
    params.pop('eval_metric', None)
    factories = {
        'DT'  : lambda: DecisionTreeClassifier(random_state=seed, **params),
        'RF'  : lambda: RandomForestClassifier(random_state=seed, **params),
        'XGB' : lambda: xgb.XGBClassifier(random_state=seed, verbosity=0, eval_metric='mlogloss', **params),
        'LGBM': lambda: lgb.LGBMClassifier(random_state=seed, verbosity=-1, **params),
        'KNN' : lambda: KNeighborsClassifier(**params),
        'MLP' : lambda: MLPClassifier(random_state=seed, **params),
    }
    return factories[clf_name]()


# ─────────────────────────────────────────────────────────────────────────────
def load_done_set(results_csv_path):
    """Set of (K, seed, classifier) tuples already completed -> resume support."""
    if os.path.exists(results_csv_path):
        existing = pd.read_csv(results_csv_path)
        if len(existing):
            return set(zip(existing['K'], existing['seed'], existing['classifier']))
    return set()


def append_result_row(results_csv_path, row):
    pd.DataFrame([row]).to_csv(
        results_csv_path, mode='a',
        header=not os.path.exists(results_csv_path),
        index=False
    )


def run_experiment_grid(method_name, transform_fn, d, results_csv_path,
                         K_values=None, seeds=None):
    """
    Generic incremental + resumable runner shared by every FS/FE method notebook.

    Parameters
    ----------
    method_name : str
        Label stored in the 'method' column (e.g. 'Correlation', 'PCA').
    transform_fn : callable(K) -> (X_tr_feat, X_val_feat, X_test_feat, actual_K)
        Must fit any transformer ONLY on training data internally.
    d : dict
        Output of load_data().
    results_csv_path : str
        CSV appended to after EVERY (K, seed, classifier) run. If the kernel
        crashes, re-running this same cell skips combinations already present
        in the file and continues from where it stopped.
    """
    K_values = K_values or K_VALUES
    seeds    = seeds or SEEDS

    done = load_done_set(results_csv_path)
    classifier_names = d['classifier_names']
    total = len(K_values) * len(seeds) * len(classifier_names)
    n_done_at_start = len(done)
    if n_done_at_start:
        print(f'[{method_name}] resuming: {n_done_at_start}/{total} combinations already done.')

    for K in K_values:
        # Skip the (expensive) transform entirely if this K is fully done
        if all((K, s, c) in done for s in seeds for c in classifier_names):
            continue

        X_tr_feat, X_val_feat, X_test_feat, actual_K = transform_fn(K)

        for seed in seeds:
            if all((K, seed, c) in done for c in classifier_names):
                continue

            X_tr_res, y_tr_res = resample_train(X_tr_feat, d['y_train'], seed)

            for clf_name in classifier_names:
                if (K, seed, clf_name) in done:
                    continue

                model = make_model(clf_name, seed, d['best_params'])
                t0 = time.perf_counter()
                model.fit(X_tr_res, y_tr_res)
                train_time = time.perf_counter() - t0

                y_pred, y_prob, inf_ms = timed_predict(model, X_test_feat)
                m = compute_metrics(d['y_test'], y_pred, d['n_classes'], y_prob)
                m['train_time_s'] = train_time
                m['inference_ms'] = inf_ms

                row = {'method': method_name, 'K': K, 'actual_K': actual_K,
                       'seed': seed, 'classifier': clf_name, **m}
                append_result_row(results_csv_path, row)
                done.add((K, seed, clf_name))

                print(f'[{method_name}] K={K} seed={seed} {clf_name:5s}  '
                      f'F1={m["f1"]:.4f}  train={train_time:.1f}s  inf={inf_ms:.4f}ms/sample  '
                      f'({len(done)}/{total})')

    print(f'\n=== {method_name}: COMPLETE -> {results_csv_path} ===')
