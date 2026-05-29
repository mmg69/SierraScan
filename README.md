# SierraScan - Sistema UniSierra de Detección de Similitud Documental

SierraScan es un sistema web desarrollado como proyecto final para la materia de **Temas de Programación Avanzada**. Su propósito es analizar similitud textual entre documentos académicos en formato PDF mediante técnicas de Procesamiento de Lenguaje Natural y un modelo supervisado de clasificación.

El sistema permite cargar uno o varios documentos PDF, extraer su contenido en texto plano, procesarlo, compararlo con otros documentos y mostrar un porcentaje de similitud. Además, integra un modelo entrenado con pares de textos etiquetados para clasificar si dos textos son similares o no similares.

---

## Descripción general

El sistema funciona mediante el siguiente flujo:

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
Características de comparación
↓
Modelo supervisado con Regresión Logística
↓
Clasificación: similar / no similar

Además, el sistema conserva el cálculo de similitud de coseno para mostrar en la interfaz un porcentaje interpretable entre documentos.

Tecnologías utilizadas
Backend
Python
FastAPI
Uvicorn
SQLite
SQLAlchemy
PyMuPDF
scikit-learn
Hugging Face Datasets
pandas
joblib
scipy
Frontend
React
Vite
TypeScript
CSS
Estructura del proyecto
Detector de Plagio/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── storage.py
│   ├── processing.py
│   ├── similarity.py
│   ├── dataset_loader.py
│   ├── evaluation.py
│   ├── train_model.py
│   ├── ml_model.py
│   └── models/
│       ├── plagiarism_pair_model.joblib
│       └── model_metrics.json
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx
│       └── App.css
├── documentacion/
│   ├── documento_final.docx
│   ├── capturas/
│   └── pdfs_prueba/
│       ├── 01_original_sistema_plagio.pdf
│       ├── 02_copia_parafraseada_plagio.pdf
│       └── 03_tema_distinto_no_plagio.pdf
├── uploads/
├── requirements.txt
├── README.md
└── .gitignore

Funcionalidades principales
Carga de uno o varios documentos PDF.
Carga mediante selección de archivos.
Carga mediante arrastrar y soltar documentos.
Extracción automática de texto desde PDF.
Normalización y tokenización del texto.
Filtrado de stopwords.
Comparación de documentos usando TF-IDF y similitud de coseno.
Visualización de porcentaje de similitud.
Clasificación del resultado en:
Similitud baja
Similitud sospechosa
Posible plagio
Entrenamiento de un modelo supervisado con TF-IDF y Regresión Logística.
Predicción de pares de texto como similares o no similares.
Consulta de métricas del modelo entrenado desde la API.
Prueba automática con dataset aleatorio.
Evaluación del sistema mediante métricas como exactitud, precisión, recall y F1-score.
Dataset utilizado

Para entrenar y validar el comportamiento del sistema se utilizó el dataset:

Mharis205/MIT-PLAGAIRISM-DETECTION-DATASET

Este dataset contiene pares de textos etiquetados como positivos o negativos:

0 = par no similar / no plagio
1 = par positivo / similar

El dataset se utilizó para entrenar el modelo supervisado y medir su desempeño. Debido a que contiene principalmente pares de oraciones cortas, también se realizaron pruebas complementarias con documentos PDF de mayor extensión.

Modelo supervisado implementado

El sistema implementa un modelo supervisado de clasificación binaria basado en:

TF-IDF + Regresión Logística

El modelo fue entrenado con pares de textos etiquetados del dataset. Cada par contiene un texto A, un texto B y una etiqueta que indica si el par es similar o no similar.

Antes del entrenamiento, los textos pasan por:

Normalización.
Tokenización.
Filtrado de stopwords.
Vectorización TF-IDF.
Extracción de características de comparación.

Las características utilizadas incluyen:

Similitud de coseno.
Similitud de Jaccard.
Coincidencia de términos.
Proporción de longitud entre textos.
Diferencias entre vectores TF-IDF.
Producto entre vectores TF-IDF.

La Regresión Logística utiliza estas características para clasificar si un par de textos es similar o no similar.

Además del modelo entrenado, el sistema conserva el cálculo de similitud de coseno para mostrar en la interfaz un porcentaje interpretable entre documentos.

Rangos de interpretación

Para documentos PDF, el sistema utiliza los siguientes rangos visuales:

Porcentaje	Resultado
0% - 44.99%	Similitud baja
45% - 74.99%	Similitud sospechosa
75% - 100%	Posible plagio

Estos rangos ayudan al usuario a interpretar el porcentaje de similitud mostrado por la interfaz.

Métricas de evaluación

Para evaluar el modelo se utilizaron 4,965 pares de textos del dataset. Los datos se dividieron en ejemplos de entrenamiento y prueba.

Métrica	Resultado
Ejemplos totales	4,965
Ejemplos de entrenamiento	3,972
Ejemplos de prueba	993
Exactitud global	72.51%

Resultados por clase:

Clase	Precisión	Recall	F1-score	Soporte
No plagio	71.88%	74.04%	72.94%	497
Plagio	73.18%	70.97%	72.06%	496

Matriz de confusión:

Tipo de resultado	Cantidad
Verdaderos negativos	368
Falsos positivos	129
Falsos negativos	144
Verdaderos positivos	352

Estas métricas muestran que el modelo tuvo un desempeño equilibrado entre ambas clases.

PDFs de prueba

Dentro de la carpeta:

documentacion/pdfs_prueba/

se incluyen tres documentos utilizados para validar el sistema:

01_original_sistema_plagio.pdf: documento base.
02_copia_parafraseada_plagio.pdf: documento similar al original.
03_tema_distinto_no_plagio.pdf: documento de tema diferente.

Estos archivos permiten comprobar que el sistema detecta mayor similitud entre documentos relacionados y menor similitud entre documentos de contenido distinto.

En la prueba realizada, el sistema obtuvo aproximadamente:

Comparación	Resultado
Original vs copia parafraseada	74.22% - Similitud sospechosa
Original vs tema distinto	10.55% - Similitud baja
Copia parafraseada vs tema distinto	9.67% - Similitud baja
Instalación y ejecución
Requisitos previos

Antes de ejecutar el proyecto, instalar:

Python 3.14 o superior
Node.js
npm
Visual Studio Code u otro editor de código

Para verificar las instalaciones:

python --version
node --version
npm --version
Instalación del backend

Desde la carpeta raíz del proyecto:

cd "Detector de Plagio"

Crear entorno virtual:

python -m venv venv

Activar entorno virtual en Windows:

venv\Scripts\activate

Actualizar pip:

python -m pip install --upgrade pip

Instalar dependencias desde requirements.txt:

pip install -r requirements.txt

Ejecutar backend:

uvicorn backend.main:app --reload

El backend estará disponible en:

http://127.0.0.1:8000

La documentación automática de FastAPI estará disponible en:

http://127.0.0.1:8000/docs
Instalación alternativa del backend si falla requirements.txt

Si el comando:

pip install -r requirements.txt

marca error, se pueden instalar las dependencias manualmente con:

pip install fastapi uvicorn sqlalchemy scikit-learn python-multipart requests pymupdf datasets pandas joblib scipy

Si todavía falla, probar instalando una por una:

pip install fastapi
pip install uvicorn
pip install sqlalchemy
pip install scikit-learn
pip install python-multipart
pip install requests
pip install pymupdf
pip install datasets
pip install pandas
pip install joblib
pip install scipy

Después volver a ejecutar:

uvicorn backend.main:app --reload

También se puede ejecutar Uvicorn con:

python -m uvicorn backend.main:app --reload
Entrenamiento del modelo

El modelo ya se incluye entrenado dentro de la carpeta:

backend/models/

Sin embargo, si se desea volver a entrenarlo, se puede ejecutar el siguiente comando desde la raíz del proyecto:

python -m backend.train_model

Este comando:

Carga el dataset.
Procesa los pares de textos.
Genera características de comparación.
Entrena el modelo supervisado.
Evalúa el desempeño.
Guarda el modelo y las métricas.

Archivos generados:

backend/models/plagiarism_pair_model.joblib
backend/models/model_metrics.json

El archivo .joblib contiene el modelo entrenado, mientras que model_metrics.json guarda las métricas de evaluación.

Instalación del frontend

Abrir otra terminal y entrar a la carpeta del frontend:

cd frontend

Instalar dependencias:

npm install

Ejecutar frontend:

npm run dev

El frontend estará disponible normalmente en:

http://localhost:5173
Instalación alternativa del frontend si falla npm install

Si el comando:

npm install

falla, borrar la carpeta node_modules y el archivo package-lock.json si existen:

rmdir /s /q node_modules
del package-lock.json

Luego intentar nuevamente:

npm install

Si sigue fallando, verificar que Node.js esté instalado correctamente:

node --version
npm --version

Después ejecutar:

npm run dev
Comandos rápidos para iniciar el proyecto
Terminal 1: backend
cd "Detector de Plagio"
venv\Scripts\activate
uvicorn backend.main:app --reload

O alternativamente:

python -m uvicorn backend.main:app --reload
Terminal 2: frontend
cd "Detector de Plagio\frontend"
npm run dev
Uso del sistema
Iniciar el backend.
Iniciar el frontend.
Abrir la página web en el navegador.
Cargar uno o varios documentos PDF.
Presionar el botón de comparación.
Revisar la tabla de resultados.
Opcionalmente, ejecutar la prueba con dataset aleatorio.
Prueba recomendada con PDFs

Para validar el funcionamiento con los PDFs incluidos:

Abrir la página del sistema.
Presionar Limpiar documentos.
Cargar los tres PDFs ubicados en documentacion/pdfs_prueba/.
Presionar Comparar todos.
Revisar la tabla de resultados.

El sistema debe mostrar una similitud mayor entre el documento original y la copia parafraseada, y similitudes bajas con el documento de tema distinto.

Prueba del modelo supervisado

El backend permite consultar las métricas del modelo entrenado en:

GET /api/model-metrics

También permite probar una predicción con dos textos mediante:

POST /api/predict-pair

Ejemplo de cuerpo JSON:

{
  "text_a": "El sistema permite detectar similitud entre documentos académicos.",
  "text_b": "El sistema detecta coincidencias textuales entre documentos escolares."
}

La respuesta indica si el modelo considera el par como similar o no similar, junto con un porcentaje de confianza.

Endpoints principales
Endpoint	Método	Descripción
/api/status	GET	Verifica el estado del backend
/api/upload-pdf	POST	Carga un PDF y extrae su texto
/api/documents	GET	Lista documentos guardados
/api/documents	DELETE	Elimina todos los documentos
/api/compare	GET	Compara todos los documentos guardados
/api/load-dataset	POST	Carga ejemplos del dataset
/api/demo-random-dataset	POST	Carga y compara dataset aleatorio
/api/evaluate-dataset	GET	Evalúa el sistema con dataset
/api/find-best-threshold	GET	Busca el mejor umbral de clasificación
/api/model-metrics	GET	Devuelve las métricas del modelo supervisado entrenado
/api/predict-pair	POST	Predice si dos textos son similares o no similares usando el modelo entrenado
Base de datos

El sistema utiliza SQLite.

El archivo:

plagio.db

se genera automáticamente al iniciar el backend. No es necesario crearlo manualmente.

Este archivo almacena los documentos procesados y sus textos. Para una entrega limpia, se recomienda no subirlo a GitHub, ya que cada usuario puede generarlo nuevamente al ejecutar el proyecto.

Archivos principales del backend
Archivo	Función
main.py	Define los endpoints principales del backend con FastAPI
processing.py	Extrae, normaliza, tokeniza y filtra el texto
similarity.py	Calcula TF-IDF, similitud de coseno y clasificación interpretativa por porcentaje
dataset_loader.py	Carga y adapta ejemplos del dataset
train_model.py	Entrena el modelo supervisado con TF-IDF y Regresión Logística
ml_model.py	Carga el modelo entrenado y realiza predicciones
evaluation.py	Calcula métricas de evaluación y pruebas de umbral
storage.py	Guarda documentos y textos procesados en SQLite
models/plagiarism_pair_model.joblib	Archivo del modelo entrenado
models/model_metrics.json	Métricas generadas durante la evaluación del modelo
Archivos principales del frontend
Archivo	Función
App.tsx	Contiene la interfaz principal y la conexión con la API
App.css	Define los estilos visuales del sistema
Limitaciones
El dataset utilizado contiene principalmente pares de oraciones cortas, no documentos académicos completos.
El sistema muestra un porcentaje general de similitud, pero no identifica fragmentos específicos dentro del documento.
La versión actual no incluye autenticación de usuarios.
No genera reportes automáticos en PDF.
Las predicciones del modelo dependen de los patrones aprendidos del dataset utilizado.
Recomendaciones futuras
Probar el sistema con documentos académicos más extensos y variados.
Agregar generación automática de reportes en PDF.
Comparar documentos por secciones.
Identificar fragmentos específicos con mayor similitud.
Agregar historial de revisiones.
Implementar cuentas para docentes.
Autor

Rubén Darío Andrade Barrera
Ingeniería en Sistemas Computacionales
Materia: Temas de Programación Avanzada