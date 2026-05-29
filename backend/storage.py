
from datetime import datetime

from typing import List, Optional, Dict, Any

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker


DATABASE_URL = "sqlite:///./plagio.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


class UserDoc(Base):
    __tablename__ = "user_docs"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    raw_text = Column(Text, nullable=False)
    processed_text = Column(Text, nullable=False)
    language = Column(String(20), default="unknown")
    source_type = Column(String(50), default="upload")
    expected_label = Column(Integer, nullable=True)
    pair_group = Column(String(100), nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    """
    Crea las tablas necesarias si todavía no existen.
    """
    Base.metadata.create_all(bind=engine)


def save_doc(
    filename: str,
    raw_text: str,
    processed_text: str,
    language: str = "unknown",
    source_type: str = "upload",
    expected_label: Optional[int] = None,
    pair_group: Optional[str] = None
) -> Dict[str, Any]:
    """
    Guarda un documento en la base de datos.

    source_type:
    - upload: documento cargado por el usuario
    - dataset: documento simulado desde Hugging Face

    expected_label:
    - 1: el dataset dice que el par es plagio
    - 0: el dataset dice que el par no es plagio
    - None: no aplica para PDFs del usuario

    pair_group:
    - Sirve para relacionar los dos textos de un mismo par del dataset.
    """
    db = SessionLocal()

    try:
        doc = UserDoc(
            filename=filename,
            raw_text=raw_text[:5000],
            processed_text=processed_text,
            language=language,
            source_type=source_type,
            expected_label=expected_label,
            pair_group=pair_group
        )

        db.add(doc)
        db.commit()
        db.refresh(doc)

        return doc_to_dict(doc)

    finally:
        db.close()


def get_all_docs() -> List[Dict[str, Any]]:
    """
    Retorna todos los documentos guardados.
    """
    db = SessionLocal()

    try:
        docs = db.query(UserDoc).order_by(UserDoc.uploaded_at.desc()).all()
        return [doc_to_dict(doc) for doc in docs]

    finally:
        db.close()


def get_doc_by_id(doc_id: int) -> Optional[Dict[str, Any]]:
    """
    Busca un documento por ID.
    """
    db = SessionLocal()

    try:
        doc = db.query(UserDoc).filter(UserDoc.id == doc_id).first()

        if not doc:
            return None

        return doc_to_dict(doc)

    finally:
        db.close()


def delete_doc(doc_id: int) -> bool:
    """
    Elimina un documento por ID.
    Retorna True si lo eliminó, False si no existía.
    """
    db = SessionLocal()

    try:
        doc = db.query(UserDoc).filter(UserDoc.id == doc_id).first()

        if not doc:
            return False

        db.delete(doc)
        db.commit()

        return True

    finally:
        db.close()


def clear_all_docs() -> int:
    """
    Elimina todos los documentos.
    Retorna cuántos documentos fueron eliminados.
    """
    db = SessionLocal()

    try:
        total = db.query(UserDoc).count()
        db.query(UserDoc).delete()
        db.commit()

        return total

    finally:
        db.close()

def clear_dataset_docs() -> int:
    """
    Elimina solamente los documentos que vienen del dataset.
    No borra PDFs subidos por el usuario.
    """
    db = SessionLocal()

    try:
        total = db.query(UserDoc).filter(UserDoc.source_type == "dataset").count()

        db.query(UserDoc).filter(UserDoc.source_type == "dataset").delete()
        db.commit()

        return total

    finally:
        db.close()
def count_docs() -> int:
    """
    Cuenta cuántos documentos hay guardados.
    """
    db = SessionLocal()

    try:
        return db.query(UserDoc).count()

    finally:
        db.close()


def count_dataset_docs() -> int:
    """
    Cuenta cuántos documentos vienen del dataset.
    """
    db = SessionLocal()

    try:
        return db.query(UserDoc).filter(UserDoc.source_type == "dataset").count()

    finally:
        db.close()


def doc_to_dict(doc: UserDoc) -> Dict[str, Any]:
    """
    Convierte un objeto UserDoc en diccionario para poder retornarlo en la API.
    """
    return {
        "id": doc.id,
        "filename": doc.filename,
        "raw_text": doc.raw_text,
        "processed_text": doc.processed_text,
        "language": doc.language,
        "source_type": doc.source_type,
        "expected_label": doc.expected_label,
        "pair_group": doc.pair_group,
        "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None
    }