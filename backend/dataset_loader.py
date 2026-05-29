import re
import random
from typing import List, Dict, Any, Tuple

from datasets import load_dataset

from .processing import prepare_text_for_similarity, detect_language_simple
from .storage import save_doc


DATASET_NAME = "Mharis205/MIT-PLAGAIRISM-DETECTION-DATASET"


def load_sample_dataset(
    limit: int = 10,
    random_sample: bool = False
) -> Dict[str, Any]:
    """
    Descarga una muestra limitada del dataset de Hugging Face
    y la guarda como documentos simulados en SQLite.

    Soporta:
    - carga secuencial
    - carga aleatoria

    Cada fila normalmente viene como:
    texto1 \t texto2 \t label
    """

    if limit < 1:
        limit = 1

    if limit > 50:
        limit = 50

    # =========================================================
    # Selección del rango del dataset
    # =========================================================

    if random_sample:
        start_index = random.randint(0, 300000)
        end_index = start_index + limit

        dataset = load_dataset(
            DATASET_NAME,
            split=f"train[{start_index}:{end_index}]"
        )
    else:
        start_index = 0

        dataset = load_dataset(
            DATASET_NAME,
            split=f"train[:{limit}]"
        )

    saved_documents: List[Dict[str, Any]] = []
    skipped_rows = 0
    debug_rows = []

    # =========================================================
    # Procesamiento del dataset
    # =========================================================

    for index, row in enumerate(dataset, start=1):

        sentence1, sentence2, label = extract_pair_from_row(row)

        debug_rows.append({
            "row": index,
            "available_columns": list(row.keys()),
            "text_preview": str(row.get("text", ""))[:250],
            "sentence1_preview": sentence1[:100],
            "sentence2_preview": sentence2[:100],
            "label": label
        })

        if not sentence1.strip() or not sentence2.strip():
            skipped_rows += 1
            continue

        processed_1 = prepare_text_for_similarity(sentence1)
        processed_2 = prepare_text_for_similarity(sentence2)

        if len(processed_1.split()) < 2:
            skipped_rows += 1
            continue

        if len(processed_2.split()) < 2:
            skipped_rows += 1
            continue

        pair_group = f"dataset_pair_{index:03d}"

        doc_a = save_dataset_sentence(
            filename=f"dataset_{index:03d}_A.txt",
            text=sentence1,
            label=label,
            pair_group=pair_group
        )

        doc_b = save_dataset_sentence(
            filename=f"dataset_{index:03d}_B.txt",
            text=sentence2,
            label=label,
            pair_group=pair_group
        )

        saved_documents.append(doc_a)
        saved_documents.append(doc_b)

    # =========================================================
    # Resultado final
    # =========================================================

    return {
        "message": "Dataset de prueba cargado correctamente",
        "dataset_name": DATASET_NAME,
        "requested_pairs": limit,
        "start_index": start_index,
        "saved_documents": len(saved_documents),
        "skipped_rows": skipped_rows,
        "debug_rows": debug_rows[:3],
        "documents": saved_documents
    }


# =============================================================
# Extraer textos desde filas del dataset
# =============================================================

def extract_pair_from_row(
    row: Dict[str, Any]
) -> Tuple[str, str, int | None]:

    sentence1 = get_value(
        row,
        ["sentence1", "text1", "premise", "source", "original"]
    )

    sentence2 = get_value(
        row,
        ["sentence2", "text2", "hypothesis", "target", "suspicious"]
    )

    label = get_label(row)

    # =========================================================
    # Caso: columnas normales
    # =========================================================

    if sentence1 and sentence2:
        return sentence1, sentence2, label

    # =========================================================
    # Caso: columna única "text"
    # =========================================================

    text = str(row.get("text", "")).strip()

    if not text:
        return "", "", label

    parsed_sentence1, parsed_sentence2, parsed_label = parse_text_column(text)

    if parsed_label is not None:
        label = parsed_label

    return parsed_sentence1, parsed_sentence2, label


# =============================================================
# Parsear columna única
# =============================================================

def parse_text_column(text: str) -> Tuple[str, str, int | None]:
    """
    Obtiene:
    - texto 1
    - texto 2
    - label

    desde:
    texto1 \t texto2 \t label
    """

    parts = [
        part.strip()
        for part in text.split("\t")
        if part.strip()
    ]

    # =========================================================
    # Caso ideal:
    # texto1 \t texto2 \t label
    # =========================================================

    if len(parts) >= 3:

        first = parts[0]
        second = parts[1]

        try:
            label = int(parts[2])
        except ValueError:
            label = None

        return first, second, label

    # =========================================================
    # Caso:
    # texto1 \t texto2
    # =========================================================

    if len(parts) == 2:
        return parts[0], parts[1], None

    # =========================================================
    # Intentos alternativos
    # =========================================================

    label = extract_label_from_text(text)

    patterns = [
        r"sentence1\s*[:=]\s*(.*?)\s*sentence2\s*[:=]\s*(.*?)(?:\s*label\s*[:=]\s*[01])?$",
        r"sentence\s*1\s*[:=]\s*(.*?)\s*sentence\s*2\s*[:=]\s*(.*?)(?:\s*label\s*[:=]\s*[01])?$",
        r"text1\s*[:=]\s*(.*?)\s*text2\s*[:=]\s*(.*?)(?:\s*label\s*[:=]\s*[01])?$",
        r"original\s*[:=]\s*(.*?)\s*suspicious\s*[:=]\s*(.*?)(?:\s*label\s*[:=]\s*[01])?$",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        if match:
            first = clean_extracted_text(match.group(1))
            second = clean_extracted_text(match.group(2))

            return first, second, label

    # =========================================================
    # Separar por líneas
    # =========================================================

    useful_lines = [
        clean_extracted_text(line)
        for line in text.splitlines()
        if clean_extracted_text(line)
    ]

    if len(useful_lines) >= 2:
        return useful_lines[0], useful_lines[1], label

    # =========================================================
    # Último recurso:
    # dividir texto en dos partes
    # =========================================================

    words = text.split()

    if len(words) >= 12:

        middle = len(words) // 2

        first = " ".join(words[:middle])
        second = " ".join(words[middle:])

        return first, second, label

    return "", "", label


# =============================================================
# Extraer label desde texto
# =============================================================

def extract_label_from_text(text: str) -> int | None:

    match = re.search(
        r"label\s*[:=]\s*([01])",
        text,
        flags=re.IGNORECASE
    )

    if not match:
        return None

    return int(match.group(1))


# =============================================================
# Limpiar texto
# =============================================================

def clean_extracted_text(text: str) -> str:

    text = re.sub(
        r"label\s*[:=]\s*[01]",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"sentence\s*1\s*[:=]",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"sentence\s*2\s*[:=]",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"sentence1\s*[:=]",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"sentence2\s*[:=]",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"text1\s*[:=]",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"text2\s*[:=]",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =============================================================
# Guardar documento dataset
# =============================================================

def save_dataset_sentence(
    filename: str,
    text: str,
    label: int | None,
    pair_group: str
) -> Dict[str, Any]:

    processed_text = prepare_text_for_similarity(text)

    language = detect_language_simple(text)

    return save_doc(
        filename=filename,
        raw_text=text,
        processed_text=processed_text,
        language=language,
        source_type="dataset",
        expected_label=label,
        pair_group=pair_group
    )


# =============================================================
# Obtener valor desde posibles columnas
# =============================================================

def get_value(
    row: Dict[str, Any],
    possible_keys: List[str]
) -> str:

    for key in possible_keys:

        value = row.get(key)

        if value is not None:
            return str(value)

    return ""


# =============================================================
# Obtener label
# =============================================================

def get_label(row: Dict[str, Any]) -> int | None:

    raw_label = row.get("label")

    if raw_label is None:
        raw_label = row.get("labels")

    try:
        return int(raw_label)

    except (TypeError, ValueError):
        return None