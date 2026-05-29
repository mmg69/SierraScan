import os
import json
from typing import Dict, Any

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix

from .processing import prepare_text_for_similarity


MODEL_PATH = "backend/models/plagiarism_pair_model.joblib"
METRICS_PATH = "backend/models/model_metrics.json"


def jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(text_a.split())
    tokens_b = set(text_b.split())

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)

    return len(intersection) / len(union)


def containment_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(text_a.split())
    tokens_b = set(text_b.split())

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a.intersection(tokens_b)
    smaller = min(len(tokens_a), len(tokens_b))

    if smaller == 0:
        return 0.0

    return len(intersection) / smaller


def length_ratio(text_a: str, text_b: str) -> float:
    len_a = len(text_a.split())
    len_b = len(text_b.split())

    if len_a == 0 or len_b == 0:
        return 0.0

    return min(len_a, len_b) / max(len_a, len_b)


def load_model_bundle():
    """
    Carga el modelo entrenado desde disco.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "No se encontró el modelo entrenado. Ejecuta primero: python -m backend.train_model"
        )

    return joblib.load(MODEL_PATH)


def build_pair_features(vectorizer, text_a: str, text_b: str):
    """
    Construye las mismas características usadas durante el entrenamiento.
    """
    tfidf_a = vectorizer.transform([text_a])
    tfidf_b = vectorizer.transform([text_b])

    difference = abs(tfidf_a - tfidf_b)
    product = tfidf_a.multiply(tfidf_b)

    cosine_value = np.array(
        tfidf_a.multiply(tfidf_b).sum(axis=1)
    ).ravel()[0]

    jaccard_value = jaccard_similarity(text_a, text_b)
    containment_value = containment_similarity(text_a, text_b)
    length_ratio_value = length_ratio(text_a, text_b)

    numeric_features = np.array([
        [
            cosine_value,
            jaccard_value,
            containment_value,
            length_ratio_value
        ]
    ])

    numeric_sparse = csr_matrix(numeric_features)

    return hstack([
        difference,
        product,
        numeric_sparse
    ])


def predict_pair(text_a: str, text_b: str) -> Dict[str, Any]:
    """
    Predice si un par de textos pertenece a la clase:
    0 = No plagio / no similar
    1 = Plagio / similar
    """
    model_bundle = load_model_bundle()

    vectorizer = model_bundle["vectorizer"]
    classifier = model_bundle["classifier"]
    model_name = model_bundle.get(
        "model_name",
        "TF-IDF + Logistic Regression"
    )

    processed_a = prepare_text_for_similarity(text_a)
    processed_b = prepare_text_for_similarity(text_b)

    features = build_pair_features(
        vectorizer,
        processed_a,
        processed_b
    )

    prediction = int(classifier.predict(features)[0])

    probability = None

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(features)[0]
        probability = float(probabilities[prediction])

    return {
        "model": model_name,
        "prediction": prediction,
        "prediction_text": "Plagio / similar" if prediction == 1 else "No plagio / no similar",
        "confidence": round(probability, 4) if probability is not None else None,
        "confidence_percentage": round(probability * 100, 2) if probability is not None else None,
        "processed_text_a": processed_a,
        "processed_text_b": processed_b
    }


def get_model_metrics() -> Dict[str, Any]:
    """
    Devuelve las métricas guardadas durante el entrenamiento.
    """
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(
            "No se encontró model_metrics.json. Ejecuta primero: python -m backend.train_model"
        )

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)