Directory Structure
--------------------

    .
    ├── AUTHORS.md
    ├── LICENSE
    ├── README.md
    ├── models  <- compiled model .pkl or HDFS or .pb format
    ├── config  <- any configuration files
    ├── data
    │   ├── interim <- data in intermediate processing stage
    │   ├── processed <- data after all preprocessing has been done
    │   └── raw <- original unmodified data acting as source of truth and provenance
    ├── docs  <- usage documentation or reference papers
    ├── notebooks <- jupyter notebooks for exploratory analysis and explanation 
    ├── reports <- generated project artefacts eg. visualisations or tables
    │   └── figures
    └── src
        ├── data-proc <- scripts for processing data eg. transformations, dataset merges etc. 
        ├── viz  <- scripts for visualisation during EDA, modelling, error analysis etc. 
        ├── modeling    <- scripts for generating models
    |--- environment.yml <- file with libraries and library versions for recreating the analysis environment



Tecnología y Arquitectura
-------------------------
- **Lenguaje base**: Python 3 con un entorno reproducible definido en `environment.yml` / `requirements.txt`.
- **Procesamiento de datos**: pandas y numpy gestionan la ingestión y transformación (clases en `src/data_proc`).
- **EDA y reporting**: Matplotlib y Seaborn generan figuras; los reportes HTML están encapsulados en `EDAReport` (`src/eda`).
- **Modelado supervisado**: scikit-learn provee los algoritmos base (logistic regression, random forest, gradient boosting, SVM, k-NN) integrados vía `ModelTrainer` (`src/modeling`).
- **Orquestación**: scripts tipo CLI (`src/run_eda.py`, `src/full_pipeline.py`) coordinan los pasos; DVC puede versionar datos/artefactos (`dvc.yaml`).
- **Arquitectura**: estructura modular tipo paquete Python con separación por dominio (EDA, limpieza, modelado, pipeline) más notebooks para exploración.

EDA Workflows
-------------

Ejecutar el pipeline completo (EDA crudo + limpieza + EDA limpio):

```bash
python -m src.run_full_eda \
  --raw data/enriched/insurance_company_enriched.csv \
  --clean-out data/interim/insurance_clean.csv \
  --raw-report reports/eda_raw.html \
  --clean-report reports/eda_clean.html
```

Generar solo un reporte EDA sobre un dataset (sin limpieza):

```bash
python -m src.run_eda \
  --csv data/enriched/insurance_company_enriched.csv\
  --out reports/eda_single.html
```

Utiliza los flags opcionales (`--target`, `--id-cols`, `--datetime-cols`, `--cat-cols`, etc.) para ajustar cada comando a tus columnas objetivo o requerimientos de análisis.

Modeling (Baseline)
-------------------

La lógica del notebook `03_modeling.ipynb` ahora vive en clases reutilizables dentro de `src/modeling`.

```python
from pathlib import Path
from src.modeling import ModelConfig, ModelTrainer

# Carga parámetros desde config/params.yaml (requiere PyYAML)
config = ModelConfig.from_yaml(Path("config/params.yaml"))
trainer = ModelTrainer(config)
result = trainer.run()

print("ROC-AUC:", result.metrics["roc_auc"])
print(result.classification_report)
print(result.feature_summary.head())
```

Si no tienes PyYAML instalado, crea el `ModelConfig` manualmente:

```python
from pathlib import Path
from src.modeling import ModelConfig, ModelTrainer

config = ModelConfig(
    data_path=Path("data/interim/data_clean.parquet"),
    target="caravan",
    drop_suffixes=("_label",),
)
trainer = ModelTrainer(config)
result = trainer.run()
```

Probar otros modelos
--------------------

1. Ajusta `model.algorithm` en `config/params.yaml` (`logreg`, `rf`, `gb`, `svc`, `knn` o cualquier otra clave registrada en `MODEL_REGISTRY` dentro de `src/modeling/trainer.py`) y personaliza el bloque correspondiente (`model.logreg`, `model.rf`, `model.gb`, etc.) con los hiperparámetros que quieras evaluar.
2. Ejecuta `python3 -m src.full_pipeline --config config/params.yaml --no-report` o `dvc repro train_model`; cada corrida generará un nuevo run en MLflow con sus métricas, parámetros y artefactos. Para sobrescribir el algoritmo sin editar el YAML usa `--algorithm` (ej. `python3 -m src.full_pipeline --config config/params.yaml --no-report --algorithm rf`).
   - Desde un entorno `bash`, puedes seleccionar el modelo al vuelo:  
     ```bash
     python3 -m src.full_pipeline --config config/params.yaml --no-report --algorithm gb
     ```
   - En Windows PowerShell:  
     ```powershell
     python3 -m src.full_pipeline --config config/params.yaml --no-report --algorithm svc
     ```
3. Cambia `tracking.mlflow.run_name` o añade etiquetas en `tracking.mlflow.tags` para identificar fácilmente cada experimento (`stage: experiment`, `model_version: v2`, etc.).
4. Para incorporar algoritmos adicionales, implementa la clase en `src/modeling/models.py`, agrégala al `MODEL_REGISTRY` y vuelve a lanzar el pipeline; los resultados aparecerán en MLflow y en `models/registry`.

Ejecutar todos los modelos disponibles en secuencia (usando `bash`):

```bash
for algo in logreg rf gb svc knn; do
  python3 -m src.full_pipeline --config config/params.yaml --no-report --algorithm "$algo"
done
```

Full Pipeline
-------------

Ejecuta todo el flujo (limpieza + EDA + modelado) desde Python o CLI usando `src.full_pipeline`:

```python
from src.full_pipeline import run_full_pipeline

artifacts = run_full_pipeline()
print(artifacts["model_metrics"])
```

También puedes dispararlo desde la línea de comandos:

```bash
python -m src.full_pipeline --config config/params.yaml
```

La sección `tracking` de `config/params.yaml` controla el seguimiento automático con MLflow y el registro local de modelos:

```yaml
tracking:
  mlflow:
    enabled: true
    tracking_uri: "file:mlruns"
    experiment_name: "insurance_baseline"
    run_name: "baseline_logreg"
    registered_model_name: "InsuranceBaseline"
  registry:
    dir: "models/registry"
```

Cada ejecución del pipeline genera:
- Un run en MLflow con métricas (accuracy, precision, recall, F1, ROC-AUC, log-loss), matriz de confusión, parámetros, artefactos y el modelo serializado.
- Un historial versionado en `models/registry/<timestamp>_<algoritmo>.json` con versión, hiperparámetros, métricas y enlaces relevantes. El último resultado queda accesible en `models/registry/latest.json`.

Levanta la interfaz de MLflow para comparar configuraciones y métricas:

```bash
mlflow ui --backend-store-uri file:mlruns
```

Si ejecutas desde otra carpeta, proporciona la ruta absoluta donde se guardan los runs:

```bash
mlflow ui --backend-store-uri file:/Users/jfts/Documents/ML_OPS_PROYECT/mlops_proyect/mlruns
```

Con la UI levantada, abre `http://127.0.0.1:5000` y selecciona el experimento `insurance_baseline` para revisar las ejecuciones.

Desde la UI puedes comparar ejecuciones, descargar artefactos y visualizar las métricas registradas.

DVC Pipeline
------------

1. Instala DVC en tu entorno (`pip install dvc`).
2. Inicializa el repositorio si todavía no lo hiciste:
   ```bash
   dvc init
   ```
3. (Opcional) Versiona los datos crudos/enriquecidos con DVC:
   ```bash
   dvc add data/enriched/insurance_company_enriched.csv
   git add data/enriched/insurance_company_enriched.csv.dvc .gitignore
   ```
4. Reproduce la tubería completa de EDA y limpieza:
   ```bash
   dvc repro eda_full
   ```
   Este stage ejecuta `src.run_full_eda` y actualiza:
   - `data/interim/insurance_clean.csv`
   - `reports/eda_raw.html`
   - `reports/eda_clean.html`
   - `reports/figures/raw/`
   - `reports/figures/clean/`
5. Sincroniza artefactos con tu remoto de datos cuando sea necesario (`dvc remote add`, `dvc push`, `dvc pull`).

Consulta `dvc.yaml` para conocer dependencias y salidas de cada stage; puedes extenderlo con pasos adicionales de preprocesamiento o modelado según avances en el proyecto.

Se añadió el stage `train_model` para versionar los experimentos de modelado con DVC:

```yaml
train_model:
  cmd: python -m src.full_pipeline --config config/params.yaml --no-report
  deps:
    - config/params.yaml
    - data/enriched/insurance_company_enriched.csv
    - src/full_pipeline.py
    - src/data_proc/data_cleaner.py
    - src/modeling/config.py
    - src/modeling/models.py
    - src/modeling/trainer.py
  outs:
    - models/registry
```

Reprodúcelo con:

```bash
dvc repro train_model
```

Después, agrega `models/registry` al control de versiones (DVC o Git) y utiliza `models/registry/latest.json` como archivo de métricas para comparar ejecuciones (`dvc diff`, `dvc metrics show models/registry/latest.json`).
   
