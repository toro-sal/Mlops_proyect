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
   
