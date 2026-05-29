from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PLAGIARISM_THRESHOLD = 0.75
SUSPICIOUS_THRESHOLD = 0.45


def compare_two_texts(text_a: str, text_b: str) -> Dict[str, Any]:
    """
    Compara dos textos usando TF-IDF + similitud de coseno.

    Retorna:
    - similarity_score: valor entre 0 y 1
    - similarity_percentage: porcentaje entre 0 y 100
    - status: clasificación textual del resultado
    """
    if not text_a or not text_b:
        return {
            "similarity_score": 0.0,
            "similarity_percentage": 0.0,
            "cosine_distance": 1.0,
            "distance_percentage": 100.0,
            "status": "Texto insuficiente"
        }

    vectorizer = TfidfVectorizer()

    try:
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    except ValueError:
        score = 0.0

    percentage = round(float(score) * 100, 2)
    cosine_distance = 1 - float(score)

    return {
        "similarity_score": round(float(score), 4),
        "similarity_percentage": percentage,
        "cosine_distance": round(cosine_distance, 4),
        "distance_percentage": round(cosine_distance * 100, 2),
        "status": classify_similarity(score)
    }

    vectorizer = TfidfVectorizer()

    try:
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    except ValueError:
        score = 0.0

    percentage = round(float(score) * 100, 2)

    return {
        "similarity_score": round(float(score), 4),
        "similarity_percentage": percentage,
        "status": classify_similarity(score)
    }


def classify_similarity(score: float) -> str:
    """
    Clasifica la similitud de acuerdo con umbrales interpretables.

    >= 75%  : posible plagio
    >= 45%  : similitud sospechosa
    < 45%   : similitud baja
    """
    if score >= PLAGIARISM_THRESHOLD:
        return "Posible plagio"

    if score >= SUSPICIOUS_THRESHOLD:
        return "Similitud sospechosa"

    return "Similitud baja"


def compare_all_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compara todos los documentos entre sí.

    Ejemplo:
    doc 1 vs doc 2
    doc 1 vs doc 3
    doc 2 vs doc 3

    Nunca compara un documento consigo mismo.
    """
    results = []

    if len(docs) < 2:
        return results

    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            doc_a = docs[i]
            doc_b = docs[j]

            comparison = compare_two_texts(
                doc_a.get("processed_text", ""),
                doc_b.get("processed_text", "")
            )

            results.append({
                "doc_a_id": doc_a.get("id"),
                "doc_a_name": doc_a.get("filename"),
                "doc_b_id": doc_b.get("id"),
                "doc_b_name": doc_b.get("filename"),
                "similarity_score": comparison["similarity_score"],
                "similarity_percentage": comparison["similarity_percentage"],
                "status": comparison["status"],
                "source_a": doc_a.get("source_type"),
                "source_b": doc_b.get("source_type"),
                "expected_label_a": doc_a.get("expected_label"),
                "expected_label_b": doc_b.get("expected_label"),
                "pair_group_a": doc_a.get("pair_group"),
                "pair_group_b": doc_b.get("pair_group")
            })

    results.sort(
        key=lambda item: item["similarity_percentage"],
        reverse=True
    )

    return results


def compare_dataset_pairs_only(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compara únicamente pares del dataset que comparten el mismo pair_group.

    Esto sirve para validar si el sistema está detectando correctamente los pares
    marcados como plagio/no plagio en el dataset.
    """
    dataset_docs = [
        doc for doc in docs
        if doc.get("source_type") == "dataset" and doc.get("pair_group")
    ]

    groups = {}

    for doc in dataset_docs:
        pair_group = doc.get("pair_group")

        if pair_group not in groups:
            groups[pair_group] = []

        groups[pair_group].append(doc)

    results = []

    for pair_group, group_docs in groups.items():
        if len(group_docs) < 2:
            continue

        doc_a = group_docs[0]
        doc_b = group_docs[1]

        comparison = compare_two_texts(
            doc_a.get("processed_text", ""),
            doc_b.get("processed_text", "")
        )

        expected_label = doc_a.get("expected_label")

        results.append({
            "pair_group": pair_group,
            "doc_a_id": doc_a.get("id"),
            "doc_a_name": doc_a.get("filename"),
            "doc_b_id": doc_b.get("id"),
            "doc_b_name": doc_b.get("filename"),
            "similarity_score": comparison["similarity_score"],
            "similarity_percentage": comparison["similarity_percentage"],
            "status": comparison["status"],
            "expected_label": expected_label,
            "expected_meaning": expected_label_to_text(expected_label),
            "prediction_matches_dataset": matches_expected_label(
                comparison["similarity_score"],
                expected_label
            )
        })

    results.sort(
        key=lambda item: item["similarity_percentage"],
        reverse=True
    )

    return results


def expected_label_to_text(label: int | None) -> str:
    """
    Traduce la etiqueta original del dataset.
    1 = par positivo / similar según el dataset
    0 = par negativo / no similar según el dataset
    """
    if label == 1:
        return "Par positivo según dataset"

    if label == 0:
        return "Par negativo según dataset"

    return "Sin etiqueta"


def matches_expected_label(score: float, expected_label: int | None) -> bool | None:
    """
    Compara la predicción del sistema con la etiqueta real del dataset.

    Para simplificar:
    - Si el score es >= 75%, el sistema predice plagio.
    - Si es menor a 75%, predice no plagio.
    """
    if expected_label is None:
        return None

    predicted_label = 1 if score >= PLAGIARISM_THRESHOLD else 0

    return predicted_label == expected_label