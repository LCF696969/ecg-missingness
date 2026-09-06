"""M2：基線模型（XGBoost）與評估。

這個模組之後 M5 會重用：M5 的設計是**固定一個訓練好的模型**（train seed=0），
只變動缺失注入的 seed，所以誤差帶反映的是「缺口剛好落在哪裡」造成的變異，
而不是模型訓練的隨機性。
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

DEFAULT_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    early_stopping_rounds=30,
    tree_method="hist",
    n_jobs=-1,
)


def train_model(X_tr, y_tr, X_val, y_val, seed: int = 0, params: dict | None = None):
    """訓練一個 XGBoost，用驗證集做早停。回傳 (model, 訓練耗時秒, 最佳迭代數)。"""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    p["random_state"] = seed

    clf = XGBClassifier(**p)
    t0 = time.time()
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    elapsed = time.time() - t0
    best_it = getattr(clf, "best_iteration", None)
    return clf, elapsed, best_it


def evaluate(model, X, y) -> dict:
    """回傳 AUROC 與 AUPRC。"""
    prob = model.predict_proba(X)[:, 1]
    return {
        "auroc": float(roc_auc_score(y, prob)),
        "auprc": float(average_precision_score(y, prob)),
    }


def predict_proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
