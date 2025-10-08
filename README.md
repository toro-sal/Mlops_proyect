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
   
