import os
import json
from typing import List, Tuple

import joblib
import numpy as np
from datasets import load_dataset
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .dataset_loader import DATASET_NAME, extract_pair_from_row
from .processing import prepare_text_for_similarity


MODEL_DIR = "backend/models"
MODEL_PATH = os.path.join(MODEL_DIR, "plagiarism_pair_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")


def load_training_data(limit: int = 5000) -> Tuple[List[str], List[str], List[int]]:
    """
    Carga pares del dataset.

    X_a = texto A
    X_b = texto B
    y = etiqueta real
    """
    dataset = load_dataset(DATASET_NAME, split=f"train[:{limit}]")

    X_a = []
    X_b = []
    y = []
    skipped_rows = 0

    for row in dataset:
        text_a, text_b, label = extract_pair_from_row(row)

        if label not in [0, 1]:
            skipped_rows += 1
            continue

        if not text_a.strip() or not text_b.strip():
            skipped_rows += 1
            continue

        processed_a = prepare_text_for_similarity(text_a)
        processed_b = prepare_text_for_similarity(text_b)

        if len(processed_a.split()) < 2 or len(processed_b.split()) < 2:
            skipped_rows += 1
            continue

        X_a.append(processed_a)
        X_b.append(processed_b)
        y.append(label)

    print(f"Datos cargados: {len(y)}")
    print(f"Filas omitidas: {skipped_rows}")

    return X_a, X_b, y


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


def build_pair_features(vectorizer, texts_a: List[str], texts_b: List[str]):
    """
    Construye características para que el modelo compare pares de textos.

    Incluye:
    - diferencia absoluta entre vectores TF-IDF
    - multiplicación de vectores TF-IDF
    - similitud de coseno
    - similitud de Jaccard
    - containment
    - proporción de longitud
    """
    tfidf_a = vectorizer.transform(texts_a)
    tfidf_b = vectorizer.transform(texts_b)

    difference = abs(tfidf_a - tfidf_b)
    product = tfidf_a.multiply(tfidf_b)

    cosine_values = np.array(
        tfidf_a.multiply(tfidf_b).sum(axis=1)
    ).ravel()

    jaccard_values = np.array([
        jaccard_similarity(a, b)
        for a, b in zip(texts_a, texts_b)
    ])

    containment_values = np.array([
        containment_similarity(a, b)
        for a, b in zip(texts_a, texts_b)
    ])

    length_ratio_values = np.array([
        length_ratio(a, b)
        for a, b in zip(texts_a, texts_b)
    ])

    numeric_features = np.vstack([
        cosine_values,
        jaccard_values,
        containment_values,
        length_ratio_values
    ]).T

    numeric_sparse = csr_matrix(numeric_features)

    return hstack([
        difference,
        product,
        numeric_sparse
    ])


def train_model(limit: int = 5000) -> None:
    """
    Entrena un modelo supervisado usando:
    - TF-IDF para representar textos
    - características de comparación entre pares
    - Logistic Regression como clasificador
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_a, X_b, y = load_training_data(limit=limit)

    if len(y) < 100:
        raise ValueError("No hay suficientes datos para entrenar el modelo.")

    X_a_train, X_a_test, X_b_train, X_b_test, y_train, y_test = train_test_split(
        X_a,
        X_b,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    all_train_texts = X_a_train + X_b_train

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )

    print("Entrenando vectorizador TF-IDF...")
    vectorizer.fit(all_train_texts)

    print("Construyendo características de entrenamiento...")
    train_features = build_pair_features(
        vectorizer,
        X_a_train,
        X_b_train
    )

    print("Construyendo características de prueba...")
    test_features = build_pair_features(
        vectorizer,
        X_a_test,
        X_b_test
    )

    classifier = LogisticRegression(
        max_iter=1500,
        class_weight="balanced"
    )

    print("Entrenando modelo supervisado...")
    classifier.fit(train_features, y_train)

    print("Evaluando modelo...")
    y_pred = classifier.predict(test_features)

    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["No plagio", "Plagio"],
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

    metrics = {
        "dataset_name": DATASET_NAME,
        "total_examples": len(y),
        "train_examples": len(y_train),
        "test_examples": len(y_test),
        "model": "TF-IDF + Logistic Regression con características de comparación",
        "accuracy": round(float(accuracy), 4),
        "accuracy_percentage": round(float(accuracy) * 100, 2),
        "classes": {
            "no_plagio": {
                "precision": round(report["No plagio"]["precision"], 4),
                "recall": round(report["No plagio"]["recall"], 4),
                "f1_score": round(report["No plagio"]["f1-score"], 4),
                "support": int(report["No plagio"]["support"])
            },
            "plagio": {
                "precision": round(report["Plagio"]["precision"], 4),
                "recall": round(report["Plagio"]["recall"], 4),
                "f1_score": round(report["Plagio"]["f1-score"], 4),
                "support": int(report["Plagio"]["support"])
            }
        },
        "confusion_matrix": {
            "true_negative": int(matrix[0][0]),
            "false_positive": int(matrix[0][1]),
            "false_negative": int(matrix[1][0]),
            "true_positive": int(matrix[1][1])
        }
    }

    model_bundle = {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "model_name": metrics["model"]
    }

    joblib.dump(model_bundle, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print("\nModelo entrenado correctamente.")
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Métricas guardadas en: {METRICS_PATH}")
    print("\nResultados:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    train_model(limit=5000)