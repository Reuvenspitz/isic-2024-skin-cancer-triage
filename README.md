# Multimodal Skin Cancer Triage on ISIC 2024

This repository contains a public, cleaned version of my undergraduate data science project on the ISIC 2024 skin lesion dataset. The project studies malignant-vs-benign lesion triage under extreme class imbalance using tabular metadata, image models, and multimodal fusion.

The main artifact is a narrative notebook that combines exploratory data analysis, patient-level validation design, supervised modeling, explainability, and error analysis.

## Project Highlights

- Builds a patient-disjoint held-out split from the public training labels.
- Studies class imbalance at both lesion and patient level.
- Compares tabular models, image-only models, and late-fusion multimodal models.
- Uses a triage-oriented metric: partial AUC at TPR >= 0.80.
- Analyzes shared false-negative structure and patient-relative "ugly duckling" features.

Main held-out results reported in the notebook:

| Model family | Best held-out pAUC@TPR>=0.80 |
| --- | ---: |
| Tabular | 0.1285 |
| Image | 0.1631 |
| Fusion | 0.1689 |

## Repository Structure

```text
.
|-- notebooks/
|   `-- Multimodal Skin Cancer Triage on ISIC 2024.ipynb
|-- src/isic_triage/
|   |-- paths.py
|   |-- plotting.py
|   `-- metrics.py
|-- data/
|-- artifacts/
|-- requirements.txt
`-- README.md
```

`data/` and `artifacts/` are intentionally ignored by Git. They are local working directories for the dataset, cached features, model outputs, and generated experiment files.

## Data Setup

The ISIC 2024 dataset is not included in this repository. Place the dataset locally under `data/`, or point the notebook to another location with environment variables.

Expected default layout:

```text
data/
|-- train-metadata.csv
|-- train-image.hdf5
`-- train-image/
    `-- image/
        `-- ISIC_*.jpg
```

The notebook also supports the development layout used during the project:

```text
data/
`-- image/
    |-- train-image.hdf5
    `-- train-image/
        `-- ISIC_*.jpg
```

Optional overrides:

```bash
export ISIC_DATA_DIR=/path/to/isic-2024-data
export ISIC_ARTIFACT_DIR=/path/to/isic-2024-artifacts
```

## Running the Notebook

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Then open:

```text
notebooks/Multimodal Skin Cancer Triage on ISIC 2024.ipynb
```

Several expensive stages are controlled by flags near the top of the notebook:

```python
FORCE_RESPLIT = False
FORCE_RESCAN_IMAGE_QC = False
FORCE_RERUN_UNSUPERVISED = False
FORCE_RETRAINING = False
```

With the default settings, the notebook is intended to load cached artifacts when available. Set the flags to `True` only when you want to recompute those stages locally.

## Notes

This is an academic project and not a medical device. The analysis is intended for educational and research demonstration purposes only.
