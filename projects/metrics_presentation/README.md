# Metrics and presentation viewer

I use this small app to look through saved results from the main project and explain how the predictions connect to simulated trades. It reads saved files rather than running the research again.

## Run

From this folder:

```powershell
python -m pip install -r requirements.txt
.\run_metrics.ps1
```

Open http://localhost:8503. The entry point is `metrics_app.py`. The launcher changes to this project's directory before starting Streamlit.

The project includes English presentation pages, AAPL and AMZN summaries, worked examples, and downloadable prediction and trading statistics. Saved experiments are loaded relative to this folder, so the viewer can be copied to another location and run independently.

## Saved evidence

`artifacts/` contains copies of the two presentation experiments:

- `20261024T192312985153Z`: AAPL.
- `20261024T205450515026Z`: AMZN.

These are fixed historical snapshots. Root-level experiment tables and the weekly XGBoost 10-bps trade, position and equity ledgers needed by the viewer are included. Trained model files and other backtest ledgers are retained in the original research project rather than bundled here. Copies do not automatically update when research is rerun.

To display a newer completed experiment, copy its manifest, accuracy and performance tables into a new directory under this project's `artifacts/`. The walkthrough and sample-data pages use the two fixed experiments above.

The optional link to the research workflow expects that separate app to be running at http://localhost:8502.

