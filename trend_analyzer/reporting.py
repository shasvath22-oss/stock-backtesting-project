from pathlib import Path

import pandas as pd

from .common import complete, manifest, require


def report(run):
    run = Path(run)
    require(run,"backtest")
    m = manifest(run)
    accuracy = pd.read_csv(run/"accuracy.csv")
    # Load portfolio results separately from prediction accuracy for the report.
    perf = pd.read_csv(run/"performance.csv")
    title = "SYNTHETIC DEMO — not market evidence" if m["synthetic"] else "Historical market research"
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>Trend analyzer report</title>
<style>body{{font:15px system-ui;max-width:1400px;margin:40px auto;padding:20px}}table{{border-collapse:collapse;font-size:12px}}th,td{{padding:7px;border:1px solid #ddd}}h1{{color:#245c72}}</style></head>
<body><h1>{title}</h1><p>Run: {run.name}</p>
<p>Signals at close, execution at next open; 1/5/21-session open-to-open targets.
Fixed model settings and probability threshold; expanding-window retraining. Long/cash sleeves, zero cash yield,
zero commissions, adverse per-side slippage. Sharpe uses zero risk-free rate and 252 sessions.</p>
<h2>Predictive evaluation</h2>{accuracy.to_html(index=False,float_format=lambda x: f"{x:.4f}")}
<h2>Portfolio evaluation</h2>{perf.to_html(index=False,float_format=lambda x: f"{x:.4f}")}
<h2>Interpretation and limits</h2><p>Accuracy uses all eligible scored observations; portfolios skip overlapping
same-instrument signals. Price units are vendor split-normalized; cash distributions are credited separately.
The fixed present-day universe has survivorship bias. Test rows share market exposure and overlapping labels;
sample count is not independent evidence. No statistical significance claim is made.
Adjusted data may be revised by the vendor. This is historical research, not a live execution model.</p>
<p>Open the Streamlit app to inspect equity curves, trades, positions, folds and data coverage.</p></body></html>'''
    (run/"report.html").write_text(html,encoding="utf-8")
    complete(run,"report")
