# Lighting-Aware Adaptive Fusion for Short-Term Lighting-Load Forecasting in Smart-Building Lighting IoT

This repository contains the code prepared for the article "Lighting-Aware
Adaptive Fusion for Short-Term Lighting-Load Forecasting in Smart-Building
Lighting IoT". It provides the implementation and running script for the
short-term lighting-load forecasting experiments.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data

Create a data directory and place CU-BEMS-style CSV files under:

```text
data/11726517/2019Floor2.csv
data/11726517/2019Floor3.csv
...
data/11726517/2019Floor7.csv
```

Each file should contain `Date`, lighting load, plug load, AC load,
illuminance, temperature, and humidity columns using the original CU-BEMS
zone-column format.

## Run

```bash
bash run.sh
```

For a custom dataset with the same column format:

```bash
DATA_DIR=/path/to/data YEARS="2020" FLOORS="1 2 3" bash run.sh
```

Results are written to `outputs/experiments/`.
