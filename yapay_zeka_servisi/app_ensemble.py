"""Backward-compat shim.

Eski kod `from yapay_zeka_servisi import app_ensemble` ile import ediyordu.
Yeni kod doğrudan `ensemble_core` veya `streamlit_app` kullanmalı.
"""
from .ensemble_core import (
    ensemble_single,
    ensemble_batch,
    tfidf_predict_proba,
    bert_predict_proba_batch,
    pick_label_from_probs,
    set_stats_hook,
    ID2LABEL,
    LABEL2ID,
    LABELS,
    tokenizer,
    bert_model,
    bert_meta,
    tfidf_bundle,
    tfidf_err,
    bert_err,
    BERT_MODEL_PATH,
    TFIDF_BUNDLE_PATH,
    SERVICE_DIR,
)

__all__ = [
    "ensemble_single",
    "ensemble_batch",
    "tfidf_predict_proba",
    "bert_predict_proba_batch",
    "pick_label_from_probs",
    "set_stats_hook",
    "ID2LABEL",
    "LABEL2ID",
    "LABELS",
    "tokenizer",
    "bert_model",
    "bert_meta",
    "tfidf_bundle",
    "tfidf_err",
    "bert_err",
    "BERT_MODEL_PATH",
    "TFIDF_BUNDLE_PATH",
    "SERVICE_DIR",
]
