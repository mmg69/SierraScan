import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

type DocumentItem = {
  id: number;
  filename: string;
  source_type: string;
  expected_label: number | null;
  pair_group: string | null;
  uploaded_at: string;
};

type ComparisonResult = {
  doc_a_name: string;
  doc_b_name: string;
  similarity_percentage: number;
  status: string;
  expected_label?: number | null;
  expected_meaning?: string;
  prediction_matches_dataset?: boolean | null;
};

function App() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [results, setResults] = useState<ComparisonResult[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    if (!message) return;

    const timer = setTimeout(() => {
      setMessage("");
    }, 7000);

    return () => clearTimeout(timer);
  }, [message]);
  useEffect(() => {
    loadDocuments();
  }, []);

  async function loadDocuments() {
    const response = await fetch(`${API_URL}/api/documents`);
    const data = await response.json();
    setDocuments(data.documents || []);
  }

  async function uploadPdfs(filesToUpload: File[]) {
    const pdfFiles = filesToUpload.filter(
      (file) => file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf")
    );

    if (pdfFiles.length === 0) {
      setMessage("Selecciona o arrastra al menos un archivo PDF.");
      return;
    }

    setLoading(true);
    setMessage(`Subiendo ${pdfFiles.length} archivo(s) PDF...`);

    let uploadedCount = 0;
    let failedCount = 0;

    try {
      for (const file of pdfFiles) {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch(`${API_URL}/api/upload-pdf`, {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          uploadedCount++;
        } else {
          failedCount++;
        }
      }

      setSelectedFile(null);
      await loadDocuments();

      if (failedCount > 0) {
        setMessage(
          `Se subieron ${uploadedCount} PDF(s), pero fallaron ${failedCount}.`
        );
      } else {
        setMessage(`Se subieron correctamente ${uploadedCount} PDF(s).`);
      }
    } catch {
      setMessage("Error al conectar con el backend al subir los PDFs.");
    } finally {
      setLoading(false);
    }
  }
  function handleDragOver(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault();
  }

  function handleDrop(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault();

    const files = Array.from(event.dataTransfer.files || []);

    if (files.length === 0) {
      setMessage("No se detectó ningún archivo.");
      return;
    }

    const pdfFiles = files.filter(
      (file) => file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf")
    );

    if (pdfFiles.length === 0) {
      setMessage("Solo se permiten archivos PDF.");
      return;
    }

    if (pdfFiles.length !== files.length) {
      setMessage("Se ignoraron archivos que no eran PDF.");
    }

    setSelectedFile(pdfFiles[0]);
    uploadPdfs(pdfFiles);
  }

  async function compareAll() {
    setLoading(true);
    setMessage("Comparando documentos...");

    try {
      const response = await fetch(`${API_URL}/api/compare`);
      const data = await response.json();

      if (!response.ok) {
        setMessage(data.detail || "No se pudo comparar.");
        return;
      }

      setResults(data.results || []);
      setMessage(`Comparaciones generadas: ${data.total_comparisons}`);
    } catch {
      setMessage("Error al comparar documentos.");
    } finally {
      setLoading(false);
    }
  }


  async function clearDocuments() {
    const confirmDelete = window.confirm(
      "¿Seguro que quieres eliminar todos los documentos?"
    );

    if (!confirmDelete) return;

    setLoading(true);

    try {
      await fetch(`${API_URL}/api/documents`, {
        method: "DELETE",
      });

      setDocuments([]);
      setResults([]);
      setMessage("Documentos eliminados.");
    } catch {
      setMessage("Error al limpiar documentos.");
    } finally {
      setLoading(false);
    }
  }

  function getStatusClass(status: string) {
    const normalized = status.toLowerCase();

    if (normalized.includes("plagio")) {
      return "badge badge-danger";
    }

    if (normalized.includes("sospechosa")) {
      return "badge badge-warning";
    }

    if (normalized.includes("baja")) {
      return "badge badge-success";
    }

    return "badge badge-neutral";
  }

  function formatDatasetLabel(label?: number | null) {
    if (label === 1) return "Plagio";
    if (label === 0) return "No plagio";
    return "Sin etiqueta";
  }
  async function runRandomDatasetDemo() {
    setLoading(true);
    setMessage("Cargando y comparando dataset aleatorio...");

    try {
      const response = await fetch(`${API_URL}/api/demo-random-dataset?limit=5`, {
        method: "POST",
      });

      const data = await response.json();

      if (!response.ok) {
        setMessage(data.detail || "No se pudo ejecutar la demo aleatoria.");
        return;
      }

      setResults(data.results || []);
      setMessage(
        `Demo lista: ${data.load_result.saved_documents} documentos cargados y ${data.total_results} pares comparados.`
      );

      await loadDocuments();
    } catch {
      setMessage("Error al ejecutar la demo aleatoria.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app">
      <section className="hero">
        <p className="eyebrow">Universidad de la Sierra</p>
        <h1>SierraScan</h1>
        <p>
          Sistema UniSierra de detección de similitud documental para analizar
          documentos PDF convertidos a texto plano y mostrar posibles coincidencias.
        </p>
      </section>

      <section className="panel">
        <h2>Panel de análisis documental</h2>

        <div className="actions">
          <button onClick={runRandomDatasetDemo} disabled={loading}>
            Probar con dataset aleatorio
          </button>

          <label className="file-input">
            <span>Seleccionar PDFs</span>
            <input
              type="file"
              accept="application/pdf"
              multiple
              onChange={(event) => {
                const files = Array.from(event.target.files || []);

                if (files.length > 0) {
                  setSelectedFile(files[0]);
                  uploadPdfs(files);
                  event.target.value = "";
                }
              }}
            />
          </label>

          <button onClick={compareAll} disabled={loading}>
            Comparar todos
          </button>

          <button className="secondary" onClick={clearDocuments} disabled={loading}>
            Limpiar documentos
          </button>
        </div>

        <div
          className="drop-zone"
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          <strong>Arrastra uno o varios PDFs aquí</strong>
          <span>también puedes usar el botón de selección múltiple</span>
        </div>
        {selectedFile && (
          <p className="selected-file">
            Último archivo procesado: {selectedFile.name}
          </p>
        )}

        {message && <p className="message">{message}</p>}
      </section>

      <section className="grid">
        <div className="panel">
          <h2>Documentos guardados</h2>
          <p className="counter">{documents.length} documentos</p>

          <div className="doc-list">
            {documents.map((doc) => (
              <article key={doc.id} className="doc-card">
                <strong>{doc.filename}</strong>
                <span>Origen: {doc.source_type}</span>
                <span>Etiqueta: {formatDatasetLabel(doc.expected_label)}</span>
              </article>
            ))}
          </div>
        </div>

        <div className="panel">
          <h2>Resultados de similitud</h2>

          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Documento A</th>
                  <th>Documento B</th>
                  <th>Similitud</th>
                  <th>Resultado</th>
                </tr>
              </thead>
              <tbody>
                {results.map((result, index) => (
                  <tr key={index}>
                    <td>{result.doc_a_name}</td>
                    <td>{result.doc_b_name}</td>
                    <td>{result.similarity_percentage}%</td>
                    <td>
                      <span className={getStatusClass(result.status)}>
                        {result.status}
                      </span>
                    </td>
                  </tr>
                ))}

                {results.length === 0 && (
                  <tr>
                    <td colSpan={4} className="empty">
                      Todavía no hay resultados de comparación.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </main>
  );
}

export default App;