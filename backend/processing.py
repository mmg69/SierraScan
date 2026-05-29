import re
import unicodedata
from typing import Optional, List

import fitz  # PyMuPDF


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extrae texto plano de un archivo PDF usando PyMuPDF.
    """
    text = ""

    try:
        with fitz.open(file_path) as pdf:
            for page in pdf:
                page_text = page.get_text("text")
                text += page_text + "\n"

    except Exception as error:
        raise ValueError(f"No se pudo extraer texto del PDF: {error}")

    return text.strip()


def tokenize_text(text: Optional[str]) -> List[str]:
    """Tokeniza el texto en palabras"""
    if not text:
        return []

    normalized = normalize_text(text)

    tokens = normalized.split()

    return tokens


def normalize_text(text: Optional[str]) -> str:
    """
    Normaliza el texto:
    - convierte a minúsculas
    - elimina acentos
    - elimina enlaces
    - elimina caracteres especiales
    - elimina espacios repetidos
    """
    if not text:
        return ""

    text = text.lower()
    text = remove_accents(text)

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9ñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_accents(text: str) -> str:
    """
    Elimina acentos del texto.
    """
    normalized = unicodedata.normalize("NFD", text)

    without_accents = "".join(
        char for char in normalized
        if unicodedata.category(char) != "Mn"
    )

    return without_accents


def get_stopwords() -> set[str]:
    """
    Lista básica de stopwords en español e inglés.

    Se usan palabras comunes que no aportan suficiente significado
    para la comparación textual.
    """
    return {
        # Español
        "el", "la", "los", "las", "un", "una", "unos", "unas",
        "de", "del", "al", "a", "ante", "bajo", "con", "contra",
        "desde", "durante", "en", "entre", "hacia", "hasta", "para",
        "por", "segun", "sin", "sobre", "tras",
        "y", "o", "u", "e", "que", "se", "su", "sus", "lo",
        "como", "mas", "pero", "porque", "ya", "si", "no",
        "es", "son", "fue", "ser", "estar", "esta", "este",
        "estos", "estas", "tambien", "donde", "cuando", "cual",
        "cuales", "quien", "quienes",

        # Inglés, útil para el dataset de Hugging Face
        "the", "a", "an", "and", "or", "but", "of", "to", "in",
        "on", "at", "for", "with", "without", "is", "are", "was",
        "were", "be", "been", "being", "this", "that", "these",
        "those", "there", "their", "his", "her", "its", "as", "by"
    }


def filter_stopwords(tokens: List[str]) -> List[str]:
    """
    Filtra stopwords de una lista de tokens.
    También elimina tokens demasiado cortos.
    """
    stopwords = get_stopwords()

    filtered_tokens = [
        token for token in tokens
        if token not in stopwords and len(token) > 2
    ]

    return filtered_tokens


def prepare_text_for_similarity(text: Optional[str]) -> str:
    """
    Prepara el texto final para el algoritmo de similitud.

    Flujo aplicado:
    1. Normalización
    2. Tokenización
    3. Filtrado de stopwords
    4. Reconstrucción del texto limpio

    Este texto limpio es el que se manda a TF-IDF.
    """
    tokens = tokenize_text(text)

    filtered_tokens = filter_stopwords(tokens)

    return " ".join(filtered_tokens)


def detect_language_simple(text: str) -> str:
    """
    Detección simple de idioma.
    No es una detección profesional, pero sirve para clasificar textos básicos.
    """
    text_lower = text.lower()

    spanish_markers = [
        "el", "la", "los", "las", "que", "para", "con", "documento",
        "sistema", "investigacion", "academico", "universidad"
    ]

    english_markers = [
        "the", "and", "for", "with", "document", "system",
        "research", "academic", "university", "person", "children"
    ]

    spanish_score = sum(
        1 for word in spanish_markers
        if f" {word} " in f" {text_lower} "
    )

    english_score = sum(
        1 for word in english_markers
        if f" {word} " in f" {text_lower} "
    )

    if spanish_score > english_score:
        return "spanish"

    if english_score > spanish_score:
        return "english"

    return "unknown"


def is_text_valid_for_analysis(text: Optional[str]) -> bool:
    """
    Verifica si el texto tiene suficiente contenido para analizarse.
    Esta validación se usa principalmente para PDFs.
    """
    if not text:
        return False

    tokens = tokenize_text(text)

    filtered_tokens = filter_stopwords(tokens)

    return len(filtered_tokens) >= 10