# DRIFT-SENSE: Wafer Drift Recovery Engine

## Quickstart

1. Generate SEM Dataset:
   python generate_dataset.py --train-samples 500 --val-samples 100 --out-dir dataset

2. Run Official Benchmark Evaluation:
   python evaluate.py --csv dataset/val/labels.csv

3. Launch Interactive UI (Optional):
   streamlit run app.py