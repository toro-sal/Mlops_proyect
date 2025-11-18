# Proyecto MLOps – Insurance Caravan

Este repositorio implementa un flujo MLOps completo para predecir la probabilidad de que un cliente tenga pólizas tipo **CARAVAN** (casas rodantes). Incluye limpieza de datos, EDA, entrenamiento con selección automática de modelos, monitoreo de drift, API de predicción (FastAPI) y empaquetado Docker.

---

## 1. Requisitos previos

- Python 3.11 (recomendado) + `pip`
- Git, DVC (opcional) y Docker (para despliegue containerizado)
- MLflow (se instala con `requirements.txt`)

Instalación rápida:

```bash
cd mlops_proyect
python -m venv ../venv && source ../venv/bin/activate
pip install -r requirements.txt
```

(Con Conda: `conda env create -f environment.yml && conda activate mlops_proyect`).

---

## 2. Estructura del repositorio

```
mlops_proyect/
├── config/                 # Parámetros globales (params.yaml, etc.)
├── data/
│   ├── enriched/           # Dataset crudo usado como entrada
│   ├── interim/            # Salidas intermedias (csv/parquet limpios)
│   └── processed/          # Features listas para modelado (si aplica)
├── models/
│   └── registry/           # Historial de modelos (JSON + .joblib + latest.json)
├── reports/
│   ├── eda_*.html          # Reportes EDA
│   ├── grid_search/        # Resultados de barridos
│   └── drift/              # Reportes de data drift
├── src/
│   ├── api/                # Servicio FastAPI (`src/api/app.py`)
│   ├── data_proc/          # Limpieza y utilidades de datos
│   ├── eda/                # Reportes exploratorios
│   ├── modeling/           # Config, modelos sklearn, grid search
│   ├── monitoring/         # Scripts de drift
│   └── full_pipeline.py    # Orquestador principal
├── tests/                  # Pytest (unitarias + integración + API)
├── dvc.yaml                # Pipeline reproducible (EDA, entrenamiento, drift, grid)
├── Dockerfile              # Imagen para servir la API
└── README.md               # Este documento
```

---

## 3. Configuración clave (`config/params.yaml`)

- `paths`: ubicación de datos crudos, limpios, reportes y modelos.
- `storage`: banderas para sincronizar automáticamente con S3/DVC (`storage.s3`, `storage.dvc`).
- `eda`: columnas ID, categóricas, objetivo, títulos de reporte, etc.
- `clean`: reglas de limpieza (threshold de NA, outliers, coerción numérica).
- `preprocess`: imputaciones, escalado, split train/test.
- `model`: algoritmo por defecto y bloques con hiperparámetros (logreg, rf, gb, svc, knn…).
- `tracking`: ajustes de MLflow (`run_name`, etiquetas, URI) y directorio del registry local.
- `model_selection`: lista de algoritmos y grids evaluados automáticamente en el pipeline.
- `monitoring`: umbrales para alertar drift (`roc_auc_drop_threshold`, severidad, etc.).

Modificar este archivo es la forma principal de adaptar el proyecto a nuevos datos/objetivos.

---

## 4. Paso a paso del pipeline

1. **Preparar entorno**  
   - Activar el venv/conda.
   - Validar datos crudos: `ls data/enriched/insurance_company_enriched.csv`.

2. **EDA (opcional pero recomendado)**  
   - Completo (crudo → limpio → reportes):
     ```bash
     python -m src.run_full_eda \
       --raw data/enriched/insurance_company_enriched.csv \
       --clean-out data/interim/insurance_clean.csv \
       --raw-report reports/eda_raw.html \
       --clean-report reports/eda_clean.html
     ```
   - Rápido (un solo dataset):
     ```bash
     python -m src.run_eda \
       --csv data/enriched/insurance_company_enriched.csv \
       --out reports/eda_single.html
     ```

3. **Pipeline completo con selección automática**  
   ```bash
   python -m src.full_pipeline --config config/params.yaml --no-report
   ```
   - Limpia datos, genera parquet intermedio y ejecuta grid search declarado en `model_selection`.
   - Cada algoritmo se reentrena únicamente con su mejor combinación y se registra en MLflow con run name `"<run_name>_<algoritmo>_best"` (ej: `insurance_run_rf_best`).
   - El mejor score global (según `model.scoring`, por defecto ROC-AUC) se publica en `models/registry/latest.json`, listo para servir en la API.
   - Para desactivar la selección y entrenar sólo `model.algorithm`: añade `--no-model-selection`.
   - Para forzar un algoritmo puntual: `--algorithm rf`.

4. **Revisar resultados en MLflow**  
   ```bash
   mlflow ui --backend-store-uri file:mlruns
   ```
   Abrir `http://127.0.0.1:5000` y filtrar por etiquetas (`model.algorithm`, etc.).

5. **Revisión manual de grid search (opcional)**  
   ```bash
   python -m src.modeling.grid_search --config config/params.yaml --output-dir reports/grid_search
   ```
   Archivos:
   - `grid_search_results.csv|json`: todas las combinaciones evaluadas.
   - `best_model.json`: mejor combinación global.
   - `best_by_algorithm.json`: campeón por modelo.
   - `best_by_algorithm_registered.json`: mismas entradas pero reentrenadas/logueadas (artefactos listos para servir). Usa `--no-register-best` si sólo quieres métricas.

6. **Servicio FastAPI**  
   ```bash
   cd mlops_proyect
   source ../venv/bin/activate
   uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
   ```
   - Endpoints:
     - `GET /health`: estado del modelo cargado desde `models/registry/latest.json`.
     - `POST /predict`: acepta `{"records": [...], "return_probabilities": true}` y responde:
       ```json
       {
         "model_version": "...",
         "algorithm": "knn",
         "predictions": [0],
         "probabilities": [0.21],
         "interpretations": ["Baja probabilidad de póliza caravan (confianza 21.0%)"],
         "feature_names": [...],
         "reference_metrics": {...}
       }
       ```
     - Variables opcionales: `MODEL_REGISTRY_METADATA_PATH`, `MODEL_ARTIFACT_PATH`, `MODEL_SERVICE_FILL_VALUE`.
   - Prueba con Postman/Curl:
     ```bash
     curl -X POST http://localhost:8000/predict \
       -H "Content-Type: application/json" \
       -d '{"records":[{"MOSTYPE":11,"MOSHOOFD":0,"MRELGE":1,"ABRAND":1,"AFIETS":1,...}],"return_probabilities":true}'
     ```

7. **Docker (servir sin entorno local)**  
   ```bash
   docker build -t ml-service:latest .
   docker run -p 8000:8000 -v $(pwd)/models:/app/models ml-service:latest
   ```
   - Usa `MODEL_REGISTRY_METADATA_PATH=/app/models/registry/latest.json` dentro del contenedor.

8. **Monitoreo de Data Drift**  
   ```bash
   python -m src.monitoring.data_drift \
     --config config/params.yaml \
     --registry models/registry/latest.json \
     --out reports/drift \
     --severity 0.3
   ```
   - Salidas: `drift_report.json`, `drift_sample.csv`, `feature_shift.png`.
   - Si `roc_auc_drop` supera `monitoring.drift.roc_auc_drop_threshold`, planificar reentrenamiento.
   - Stage DVC: `dvc repro detect_drift`.

9. **DVC (pipeline reproducible)**  
   - `dvc repro eda_full`
   - `dvc repro train_model`
   - `dvc repro grid_search`
   - `dvc repro detect_drift`
   Cada stage usa `dvc.yaml` y guarda artefactos en `models/registry`, `reports/*`, etc.

10. **Pruebas automáticas**  
    ```bash
    python -m pytest -q
    ```
    - `tests/test_full_pipeline_integration.py` valida que el pipeline genere `models/registry/latest.json`.
    - `tests/test_api.py` levanta el servicio y prueba `POST /predict`.

---

## 5. FAQ / Problemas comunes

| Problema | Solución |
| --- | --- |
| `ModuleNotFoundError: src.api` al arrancar Uvicorn | Asegúrate de ejecutar `uvicorn` desde `mlops_proyect/` o exporta `PYTHONPATH` apuntando a esa carpeta. |
| `pytest` no existe | Instala dependencias (`pip install -r requirements.txt`) dentro del entorno activo. |
| API no encuentra el modelo | Verifica que `models/registry/latest.json` exista (ejecutar pipeline), o exporta `MODEL_REGISTRY_METADATA_PATH` apuntando al archivo correcto. |
| Quiero servir un modelo específico | Usa `MODEL_ARTIFACT_PATH=/ruta/a/mi_modelo.joblib` antes de ejecutar Uvicorn. |
| Necesito sólo EDA | Ejecuta `python -m src.run_eda ...` o `dvc repro eda_full` y omite el pipeline. |

---

## 6. Resumen rápido de comandos

| Acción | Comando |
| --- | --- |
| Instalar deps | `pip install -r requirements.txt` |
| EDA completo | `python -m src.run_full_eda --raw ...` |
| Pipeline + selección | `python -m src.full_pipeline --config config/params.yaml --no-report` |
| MLflow UI | `mlflow ui --backend-store-uri file:mlruns` |

---

## 7. Integración con S3 + versionamiento DVC

1. **Configurar credenciales de AWS de forma segura**
   ```bash
   aws configure  # guarda AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY en ~/.aws/credentials
   # o usar variables de entorno:
   export AWS_ACCESS_KEY_ID=...
   export AWS_SECRET_ACCESS_KEY=...
   export AWS_SESSION_TOKEN=...   # si aplica
   ```
   > Nunca hardcodees llaves en el repositorio. `boto3` detecta automáticamente variables de entorno, perfiles y roles de IAM.

2. **Activar la sincronización en `config/params.yaml`**
   ```yaml
   storage:
     s3:
       enabled: true
       bucket: itesm-mna
       region: us-east-2
       prefix: 202502-equipo28
       artifacts:
         raw: 202502-equipo28/raw/insurance_company_enriched.csv   # opcional
     dvc:
       auto_push: true
       remote: s3-prod
   ```
   - `storage.s3.enabled`: descarga el raw si no existe y sube `raw`, `clean`, reportes y `models/registry` al bucket tras cada ejecución del stage.
   - `storage.dvc.auto_push`: ejecuta `dvc push <stage>` automáticamente (usa los nombres de `storage.dvc.stage_map` para cada etapa declarada en `dvc.yaml`).

3. **Stage → comando DVC sugerido**

| Stage (`dvc.yaml`) | Pipeline/Script | Comando para publicar versión |
| --- | --- | --- |
| `eda_full` | `python -m src.run_full_eda ...` | `dvc push eda_full -r s3-prod` |
| `train_model` | `python -m src.full_pipeline ...` | `dvc push train_model -r s3-prod` |
| `grid_search` | `python -m src.modeling.grid_search ...` | `dvc push grid_search -r s3-prod` |
| `detect_drift` | `python -m src.monitoring.data_drift ...` | `dvc push detect_drift -r s3-prod` |

Con esto tienes el dataset y los artefactos versionados en S3 sin exponer credenciales ni duplicar lógica en notebooks.
| Grid search manual | `python -m src.modeling.grid_search --config config/params.yaml --output-dir reports/grid_search` |
| API local | `uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000` |
| Docker run | `docker run -p 8000:8000 -v $(pwd)/models:/app/models ml-service:latest` |
| Drift | `python -m src.monitoring.data_drift --config config/params.yaml --registry models/registry/latest.json --out reports/drift` |
| Tests | `python -m pytest -q` |

Con este README tienes la **estructura**, el **paso a paso** y las **explicaciones** necesarias para reproducir el flujo completo, evaluar modelos y servir el mejor candidato mediante la API o Docker. ¡Listo para operar en cualquier entorno!***
