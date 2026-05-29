# SierraScan - Sistema UniSierra de Detección de Similitud Documental

SierraScan es un sistema web desarrollado como proyecto final para la materia de **Temas de Programación Avanzada**. Su propósito es analizar similitud textual entre documentos académicos en formato PDF mediante técnicas de Procesamiento de Lenguaje Natural.

El sistema permite cargar uno o varios documentos PDF, extraer su contenido en texto plano, procesarlo, compararlo con otros documentos y mostrar un porcentaje de similitud. También incluye una prueba automática con dataset etiquetado para validar el comportamiento del algoritmo.

---

## Descripción general

El sistema funciona mediante el siguiente flujo:

```txt
Documento PDF
↓
Extracción de texto plano
↓
Normalización del texto
↓
Tokenización
↓
Filtrado de stopwords
↓
Vectorización TF-IDF
↓
Similitud de coseno
↓
Porcentaje de similitud
```

El resultado se muestra en una tabla donde se indica qué documentos fueron comparados, el porcentaje de similitud obtenido y una clasificación del resultado.

---

## Tecnologías utilizadas

### Backend

* Python
* FastAPI
* Uvicorn
* SQLite
* SQLAlchemy
* PyMuPDF
* scikit-learn
* Hugging Face Datasets
* pandas

### Frontend

* React
* Vite
* TypeScript
* CSS

---

## Estructura del proyecto

```txt
Detector de Plagio/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── storage.py
│   ├── processing.py
│   ├── similarity.py
│   ├── dataset_loader.py
│   └── evaluation.py
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx
│       └── App.css
├── documentacion/
│   ├── documento_final.docx
│   ├── capturas/
│   │   ├── interfaz_general.png
│   │   ├── resultados_pdf.png
│   │   ├── fastapi_docs.png
│   │   └── metricas_modelo.png
│   └── pdfs_prueba/
│       ├── 01_original_sistema_plagio.pdf
│       ├── 02_copia_parafraseada_plagio.pdf
│       └── 03_tema_distinto_no_plagio.pdf
├── uploads/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Funcionalidades principales

* Carga de uno o varios documentos PDF.
* Carga mediante selección de archivos.
* Carga mediante arrastrar y soltar documentos.
* Extracción automática de texto desde PDF.
* Normalización y tokenización del texto.
* Filtrado de stopwords.
* Comparación de documentos usando TF-IDF y similitud de coseno.
* Visualización de porcentaje de similitud.
* Clasificación del resultado en:

  * Similitud baja
  * Similitud sospechosa
  * Posible plagio
* Prueba automática con dataset aleatorio.
* Evaluación del sistema mediante métricas como exactitud, precisión, recall y F1-score.

---

## Dataset utilizado

Para validar el comportamiento del sistema se utilizó el dataset:

```txt
Mharis205/MIT-PLAGAIRISM-DETECTION-DATASET
```

Este dataset contiene pares de textos etiquetados como positivos o negativos:

* `0` = par no similar / no plagio
* `1` = par positivo / similar

El dataset se utilizó como mecanismo de validación inicial, ya que contiene principalmente pares de oraciones cortas. El sistema final está orientado principalmente al análisis de documentos académicos más extensos en formato PDF.

---

## Modelo de similitud textual

El sistema utiliza un modelo de similitud textual basado en:

```txt
TF-IDF + Similitud de coseno
```

TF-IDF convierte los textos en vectores numéricos, asignando mayor importancia a palabras relevantes y menor importancia a palabras comunes. Después, la similitud de coseno compara los vectores para obtener un porcentaje de semejanza entre documentos.

El sistema no utiliza un clasificador supervisado tradicional como KNN, árbol de decisión o SVM. En su lugar, utiliza un enfoque de comparación textual y clasificación por umbrales.

---

## Rangos de interpretación

Para documentos PDF, el sistema utiliza los siguientes rangos:

|   Porcentaje | Resultado            |
| -----------: | -------------------- |
|  0% - 44.99% | Similitud baja       |
| 45% - 74.99% | Similitud sospechosa |
|   75% - 100% | Posible plagio       |

---

## Métricas de evaluación

Para evaluar el comportamiento del sistema se analizaron 200 pares de textos del dataset. El mejor umbral obtenido para esta prueba fue de 17%.

| Métrica              | Resultado |
| -------------------- | --------: |
| Pares evaluados      |       200 |
| Umbral de evaluación |       17% |
| Exactitud global     |       65% |

Para la clase positiva o “Plagio”, el sistema obtuvo:

| Métrica   | Resultado |
| --------- | --------: |
| Precisión |    63.64% |
| Recall    |       70% |
| F1-score  |    66.67% |

Estos resultados corresponden a una evaluación inicial con pares de oraciones cortas. Por ello, también se realizaron pruebas complementarias con documentos PDF de mayor extensión.

---

## PDFs de prueba

Dentro de la carpeta:

```txt
documentacion/pdfs_prueba/
```

se incluyen tres documentos utilizados para validar el sistema:

* `01_original_sistema_plagio.pdf`: documento base.
* `02_copia_parafraseada_plagio.pdf`: documento similar al original.
* `03_tema_distinto_no_plagio.pdf`: documento de tema diferente.

Estos archivos permiten comprobar que el sistema detecta mayor similitud entre documentos relacionados y menor similitud entre documentos de contenido distinto.

En la prueba realizada, el sistema obtuvo aproximadamente:

| Comparación                         |   Resultado esperado |
| ----------------------------------- | -------------------: |
| Original vs copia parafraseada      | Similitud sospechosa |
| Original vs tema distinto           |       Similitud baja |
| Copia parafraseada vs tema distinto |       Similitud baja |

---

# Instalación y ejecución

## Requisitos previos

Antes de ejecutar el proyecto, instalar:

* Python 3.14 o superior
* Node.js
* npm
* Visual Studio Code u otro editor de código

Para verificar las instalaciones:

```bash
python --version
node --version
npm --version
```

---

## Instalación del backend

Desde la carpeta raíz del proyecto:

```bash
cd "Detector de Plagio"
```

Crear entorno virtual:

```bash
python -m venv venv
```

Activar entorno virtual en Windows:

```bash
venv\Scripts\activate
```

Actualizar pip:

```bash
python -m pip install --upgrade pip
```

Instalar dependencias desde `requirements.txt`:

```bash
pip install -r requirements.txt
```

Ejecutar backend:

```bash
uvicorn backend.main:app --reload
```

El backend estará disponible en:

```txt
http://127.0.0.1:8000
```

La documentación automática de FastAPI estará disponible en:

```txt
http://127.0.0.1:8000/docs
```

---

## Instalación alternativa del backend si falla `requirements.txt`

Si el comando:

```bash
pip install -r requirements.txt
```

marca error, se pueden instalar las dependencias manualmente con:

```bash
pip install fastapi uvicorn sqlalchemy scikit-learn python-multipart requests pymupdf datasets pandas
```

Si todavía falla, probar instalando una por una:

```bash
pip install fastapi
pip install uvicorn
pip install sqlalchemy
pip install scikit-learn
pip install python-multipart
pip install requests
pip install pymupdf
pip install datasets
pip install pandas
```

Después volver a ejecutar:

```bash
uvicorn backend.main:app --reload
```

---

## Problemas comunes del backend

### 1. No reconoce `uvicorn`

Si aparece un error indicando que `uvicorn` no se reconoce, verificar que el entorno virtual esté activado:

```bash
venv\Scripts\activate
```

Luego instalarlo manualmente:

```bash
pip install uvicorn
```

También se puede ejecutar con:

```bash
python -m uvicorn backend.main:app --reload
```

---

### 2. Error con PyMuPDF o `fitz`

Si aparece un error relacionado con:

```txt
fitz
```

instalar PyMuPDF:

```bash
pip install pymupdf
```

---

### 3. Error al cargar dataset

Si aparece un error relacionado con Hugging Face o datasets:

```bash
pip install datasets pandas
```

Después reiniciar el backend.

---

### 4. El archivo `plagio.db` no existe

No es un error. La base de datos SQLite se genera automáticamente al iniciar el backend.

Ejecutar:

```bash
uvicorn backend.main:app --reload
```

---

## Instalación del frontend

Abrir otra terminal y entrar a la carpeta del frontend:

```bash
cd frontend
```

Instalar dependencias:

```bash
npm install
```

Ejecutar frontend:

```bash
npm run dev
```

El frontend estará disponible normalmente en:

```txt
http://localhost:5173
```

---

## Instalación alternativa del frontend si falla `npm install`

Si el comando:

```bash
npm install
```

falla, borrar la carpeta `node_modules` y el archivo `package-lock.json` si existen:

```bash
rmdir /s /q node_modules
del package-lock.json
```

Luego intentar nuevamente:

```bash
npm install
```

Si sigue fallando, verificar que Node.js esté instalado correctamente:

```bash
node --version
npm --version
```

Después ejecutar:

```bash
npm run dev
```

---

## Comandos rápidos para iniciar el proyecto

### Terminal 1: backend

```bash
cd "Detector de Plagio"
venv\Scripts\activate
uvicorn backend.main:app --reload
```

O alternativamente:

```bash
python -m uvicorn backend.main:app --reload
```

### Terminal 2: frontend

```bash
cd "Detector de Plagio\frontend"
npm run dev
```

---

## Uso del sistema

1. Iniciar el backend.
2. Iniciar el frontend.
3. Abrir la página web en el navegador.
4. Cargar uno o varios documentos PDF.
5. Presionar el botón de comparación.
6. Revisar la tabla de resultados.
7. Opcionalmente, ejecutar la prueba con dataset aleatorio.

---

## Prueba recomendada con PDFs

Para validar el funcionamiento con los PDFs incluidos:

1. Abrir la página del sistema.
2. Presionar **Limpiar documentos**.
3. Cargar los tres PDFs ubicados en `documentacion/pdfs_prueba/`.
4. Presionar **Comparar todos**.
5. Revisar la tabla de resultados.

El sistema debe mostrar una similitud mayor entre el documento original y la copia parafraseada, y similitudes bajas con el documento de tema distinto.

---

## Endpoints principales

| Endpoint                   | Método | Descripción                            |
| -------------------------- | ------ | -------------------------------------- |
| `/api/status`              | GET    | Verifica el estado del backend         |
| `/api/upload-pdf`          | POST   | Carga un PDF y extrae su texto         |
| `/api/documents`           | GET    | Lista documentos guardados             |
| `/api/documents`           | DELETE | Elimina todos los documentos           |
| `/api/compare`             | GET    | Compara todos los documentos guardados |
| `/api/load-dataset`        | POST   | Carga ejemplos del dataset             |
| `/api/demo-random-dataset` | POST   | Carga y compara dataset aleatorio      |
| `/api/evaluate-dataset`    | GET    | Evalúa el sistema con dataset          |
| `/api/find-best-threshold` | GET    | Busca el mejor umbral de clasificación |

---

## Base de datos

El sistema utiliza SQLite.

El archivo:

```txt
plagio.db
```

se genera automáticamente al iniciar el backend. No es necesario crearlo manualmente.

Este archivo almacena los documentos procesados y sus textos. Para una entrega limpia, se recomienda no subirlo a GitHub, ya que cada usuario puede generarlo nuevamente al ejecutar el proyecto.

---

## Archivos principales del backend

| Archivo             | Función                                             |
| ------------------- | --------------------------------------------------- |
| `main.py`           | Define los endpoints de FastAPI                     |
| `processing.py`     | Extrae, normaliza, tokeniza y filtra texto          |
| `similarity.py`     | Calcula TF-IDF, similitud de coseno y clasificación |
| `dataset_loader.py` | Carga pares del dataset                             |
| `evaluation.py`     | Calcula métricas del sistema                        |
| `storage.py`        | Maneja la base de datos SQLite                      |

---

## Archivos principales del frontend

| Archivo   | Función                                                 |
| --------- | ------------------------------------------------------- |
| `App.tsx` | Contiene la interfaz principal y la conexión con la API |
| `App.css` | Define los estilos visuales del sistema                 |

---

## Limitaciones

* El sistema compara principalmente coincidencias léxicas, por lo que puede tener limitaciones ante paráfrasis profundas.
* El dataset utilizado contiene pares de oraciones cortas, no documentos académicos completos.
* La versión actual muestra un porcentaje general de similitud, pero no identifica fragmentos específicos dentro del documento.
* No incluye autenticación de usuarios.
* No genera reportes automáticos en PDF.

---

## Recomendaciones futuras

* Probar el sistema con documentos académicos más extensos y variados.
* Agregar generación automática de reportes en PDF.
* Comparar documentos por secciones.
* Identificar fragmentos específicos con mayor similitud.
* Agregar historial de revisiones.
* Implementar cuentas para docentes.

---

## Autor

**Rubén Darío Andrade Barrera**
Ingeniería en Sistemas Computacionales
Universidad de la Sierra
Materia: Temas de Programación Avanzada
