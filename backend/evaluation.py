from typing import Dict, Any, List

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from .dataset_loader import DATASET_NAME, extract_pair_from_row
from .processing import prepare_text_for_similarity
from .similarity import compare_two_texts


def evaluate_dataset_model(
    limit: int = 200,
    threshold: float = 0.45
) -> Dict[str, Any]:
    """
    Evalúa el comportamiento del sistema usando el dataset etiquetado.

    label real del dataset:
    - 1 = par positivo / plagio / similitud esperada
    - 0 = par negativo / no plagio

    predicción del sistema:
    - 1 si similitud >= threshold
    - 0 si similitud < threshold
    """

    if limit < 10:
        limit = 10

    if limit > 1000:
        limit = 1000

    dataset = load_dataset(DATASET_NAME, split=f"train[:{limit}]")

    y_true: List[int] = []
    y_pred: List[int] = []
    evaluated_examples = []
    skipped_rows = 0

    for index, row in enumerate(dataset, start=1):
        text_a, text_b, label = extract_pair_from_row(row)

        if label is None:
            skipped_rows += 1
            continue

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

        comparison = compare_two_texts(processed_a, processed_b)

        similarity_score = comparison["similarity_score"]

        predicted_label = 1 if similarity_score >= threshold else 0

        y_true.append(label)
        y_pred.append(predicted_label)

        if len(evaluated_examples) < 10:
            evaluated_examples.append({
                "row": index,
                "text_a": text_a,
                "text_b": text_b,
                "expected_label": label,
                "predicted_label": predicted_label,
                "similarity_percentage": comparison["similarity_percentage"],
                "status": comparison["status"]
            })

    if not y_true:
        return {
            "message": "No se pudieron evaluar ejemplos válidos.",
            "evaluated_count": 0,
            "skipped_rows": skipped_rows
        }

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0
    )

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["No plagio", "Plagio"],
        zero_division=0,
        output_dict=True
    )

    tn = int(matrix[0][0])
    fp = int(matrix[0][1])
    fn = int(matrix[1][0])
    tp = int(matrix[1][1])

    return {
        "message": "Evaluación completada correctamente",
        "dataset_name": DATASET_NAME,
        "limit_requested": limit,
        "evaluated_count": len(y_true),
        "skipped_rows": skipped_rows,
        "threshold": threshold,
        "threshold_percentage": threshold * 100,
        "accuracy": round(float(accuracy), 4),
        "accuracy_percentage": round(float(accuracy) * 100, 2),
        "classes": {
            "no_plagio": {
                "label": 0,
                "precision": round(float(precision[0]), 4),
                "recall": round(float(recall[0]), 4),
                "f1_score": round(float(f1[0]), 4),
                "support": int(support[0])
            },
            "plagio": {
                "label": 1,
                "precision": round(float(precision[1]), 4),
                "recall": round(float(recall[1]), 4),
                "f1_score": round(float(f1[1]), 4),
                "support": int(support[1])
            }
        },
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp
        },
        "classification_report": report,
        "examples": evaluated_examples
    }
def find_best_threshold(
    limit: int = 200,
    start: float = 0.10,
    end: float = 0.60,
    step: float = 0.01
) -> Dict[str, Any]:
    """
    Prueba varios umbrales de clasificación y devuelve el que obtiene
    mejor exactitud global.
    """

    results = []
    current = start

    while current <= end:
        evaluation = evaluate_dataset_model(
            limit=limit,
            threshold=round(current, 2)
        )

        if evaluation.get("evaluated_count", 0) > 0:
            results.append({
                "threshold": round(current, 2),
                "threshold_percentage": round(current * 100, 2),
                "accuracy_percentage": evaluation["accuracy_percentage"],
                "plagio_precision": round(
                    evaluation["classes"]["plagio"]["precision"] * 100,
                    2
                ),
                "plagio_recall": round(
                    evaluation["classes"]["plagio"]["recall"] * 100,
                    2
                ),
                "plagio_f1": round(
                    evaluation["classes"]["plagio"]["f1_score"] * 100,
                    2
                ),
                "no_plagio_f1": round(
                    evaluation["classes"]["no_plagio"]["f1_score"] * 100,
                    2
                )
            })

        current += step

    if not results:
        return {
            "message": "No se pudieron evaluar umbrales.",
            "results": []
        }

    best_by_accuracy = max(
        results,
        key=lambda item: item["accuracy_percentage"]
    )

    best_by_plagio_f1 = max(
        results,
        key=lambda item: item["plagio_f1"]
    )

    return {
        "message": "Búsqueda de umbral completada",
        "limit": limit,
        "range": {
            "start": start,
            "end": end,
            "step": step
        },
        "best_by_accuracy": best_by_accuracy,
        "best_by_plagio_f1": best_by_plagio_f1,
        "top_10_by_accuracy": sorted(
            results,
            key=lambda item: item["accuracy_percentage"],
            reverse=True
        )[:10],
        "results": results
    }
    