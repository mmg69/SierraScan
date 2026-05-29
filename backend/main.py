import os

from .processing import extract_text_from_pdf, prepare_text_for_similarity, detect_language_simple
from .similarity import compare_two_texts
from .dataset_loader import load_sample_dataset
from .ml_model import predict_pair, get_model_metrics 

from typing import List
from .evaluation import evaluate_dataset_model, find_best_threshold
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .storage import (
    init_db,
    save_doc,
    get_all_docs,
    get_doc_by_id,
    delete_doc,
    clear_all_docs,
    clear_dataset_docs,
    count_docs,
    count_dataset_docs
)

from .processing import (
    extract_text_from_pdf,
    prepare_text_for_similarity,
    detect_language_simple,
    is_text_valid_for_analysis
)

from .similarity import (
    compare_all_documents,
    compare_dataset_pairs_only,
    compare_two_texts
)

from .dataset_loader import load_sample_dataset


UPLOAD_DIR = "uploads"

app = FastAPI(
    title="Sistema Institucional de Detección de Plagio Documental",
    description="API para cargar documentos, procesar texto y comparar similitud textual.",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """
    Se ejecuta al iniciar el servidor.
    Crea la base de datos y la carpeta uploads si no existen.
    """
    init_db()
    os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def home():
    return {
        "message": "API del Sistema de Detección de Plagio funcionando correctamente"
    }


@app.get("/api/status")
def get_status():
    """
    Endpoint simple para verificar que el backend está activo.
    """
    return {
        "status": "ok",
        "total_documents": count_docs(),
        "dataset_documents": count_dataset_docs()
    }


@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Sube un PDF, extrae su texto, lo procesa y lo guarda en SQLite.
    """
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No se recibió ningún archivo."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Solo se permiten archivos PDF."
        )

    safe_filename = file.filename.replace(" ", "_")
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        raw_text = extract_text_from_pdf(file_path)

        if not is_text_valid_for_analysis(raw_text):
            raise HTTPException(
                status_code=400,
                detail="El PDF no contiene suficiente texto para analizar."
            )

        processed_text = prepare_text_for_similarity(raw_text)
        language = detect_language_simple(raw_text)

        saved_doc = save_doc(
            filename=safe_filename,
            raw_text=raw_text,
            processed_text=processed_text,
            language=language,
            source_type="upload"
        )

        return {
            "message": "PDF cargado y procesado correctamente",
            "document": saved_doc
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar el PDF: {error}"
        )


@app.get("/api/documents")
def list_documents():
    """
    Lista todos los documentos guardados.
    """
    documents = get_all_docs()

    return {
        "total": len(documents),
        "documents": documents
    }


@app.get("/api/documents/{doc_id}")
def read_document(doc_id: int):
    """
    Obtiene un documento específico por ID.
    """
    document = get_doc_by_id(doc_id)

    if not document:
        raise HTTPException(
            status_code=404,
            detail="Documento no encontrado."
        )

    return document


@app.delete("/api/documents/{doc_id}")
def remove_document(doc_id: int):
    """
    Elimina un documento por ID.
    """
    deleted = delete_doc(doc_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Documento no encontrado."
        )

    return {
        "message": "Documento eliminado correctamente",
        "deleted_id": doc_id
    }


@app.delete("/api/documents")
def remove_all_documents():
    """
    Elimina todos los documentos guardados.
    """
    deleted_count = clear_all_docs()

    return {
        "message": "Todos los documentos fueron eliminados",
        "deleted_count": deleted_count
    }


@app.post("/api/load-dataset")
def load_dataset_endpoint(limit: int = 10):
    """
    Carga una muestra del dataset de Hugging Face.

    Ejemplo:
    /api/load-dataset?limit=10
    """
    try:
        result = load_sample_dataset(limit=limit)
        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo cargar el dataset: {error}"
        )
@app.post("/api/demo-random-dataset")
def demo_random_dataset(limit: int = 5):
    """
    Carga un set aleatorio del dataset y lo compara automáticamente.

    Flujo:
    1. Borra documentos anteriores del dataset.
    2. Carga nuevos pares aleatorios.
    3. Compara solamente esos pares.
    4. Devuelve documentos y resultados.
    """
    try:
        deleted_dataset_docs = clear_dataset_docs()

        load_result = load_sample_dataset(
            limit=limit,
            random_sample=True
        )

        documents = get_all_docs()
        comparison_results = compare_dataset_pairs_only(documents)

        return {
            "message": "Demo aleatoria ejecutada correctamente",
            "deleted_dataset_docs": deleted_dataset_docs,
            "load_result": load_result,
            "total_results": len(comparison_results),
            "results": comparison_results
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo ejecutar la demo aleatoria: {error}"
        )

@app.get("/api/compare")
def compare_documents():
    """
    Compara todos los documentos guardados entre sí.
    """
    documents = get_all_docs()

    if len(documents) < 2:
        raise HTTPException(
            status_code=400,
            detail="Se necesitan al menos 2 documentos para comparar."
        )

    results = compare_all_documents(documents)

    return {
        "total_documents": len(documents),
        "total_comparisons": len(results),
        "results": results
    }


@app.get("/api/compare-dataset-pairs")
def compare_dataset_pairs():
    """
    Compara solamente los pares originales del dataset.
    """
    documents = get_all_docs()

    results = compare_dataset_pairs_only(documents)

    return {
        "total_results": len(results),
        "results": results
    }


@app.post("/api/compare-text")
def compare_text_directly(payload: dict):
    """
    Compara dos textos enviados directamente desde el frontend.
    Este endpoint es opcional, pero sirve para pruebas rápidas.
    """
    text_a = payload.get("text_a", "")
    text_b = payload.get("text_b", "")

    processed_a = prepare_text_for_similarity(text_a)
    processed_b = prepare_text_for_similarity(text_b)

    result = compare_two_texts(processed_a, processed_b)

    return {
        "text_a_length": len(text_a),
        "text_b_length": len(text_b),
        "result": result
    }
    
@app.get("/api/evaluate-dataset")
def evaluate_dataset(limit: int = 200, threshold: float = 0.45):
    """
    Evalúa el sistema usando el dataset etiquetado.

    Calcula:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Matriz de confusión

    threshold:
    - 0.45 significa que desde 45% de similitud se predice clase positiva.
    """
    try:
        result = evaluate_dataset_model(
            limit=limit,
            threshold=threshold
        )

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo evaluar el dataset: {error}"
        )
@app.get("/api/find-best-threshold")
def find_best_threshold_endpoint(
    limit: int = 200,
    start: float = 0.10,
    end: float = 0.60,
    step: float = 0.01
):
    """
    Prueba varios umbrales y devuelve el mejor según accuracy y F1 de plagio.
    """
    try:
        result = find_best_threshold(
            limit=limit,
            start=start,
            end=end,
            step=step
        )

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo buscar el mejor umbral: {error}"
        )
@app.get("/api/model-metrics")
def read_model_metrics():
    """
    Devuelve las métricas del modelo supervisado entrenado.
    """
    try:
        return get_model_metrics()

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudieron cargar las métricas del modelo: {error}"
        )


@app.post("/api/predict-pair")
def predict_text_pair(payload: dict):
    """
    Predice si dos textos forman un par similar/plagio usando el modelo entrenado.
    """
    text_a = payload.get("text_a", "")
    text_b = payload.get("text_b", "")

    if not text_a.strip() or not text_b.strip():
        raise HTTPException(
            status_code=400,
            detail="Se requieren text_a y text_b."
        )

    try:
        result = predict_pair(text_a, text_b)
        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo generar la predicción: {error}"
        )