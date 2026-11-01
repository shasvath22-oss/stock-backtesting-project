"""A guided view of the existing, persisted research stages."""
import pandas as pd
import streamlit as st

from .common import manifest, read_json
from .flowchart import show_flowchart


LESSONS = [
    ("collect", "Download prices", "Start with Apple's daily opening, high, low and closing prices, volume and dividends. SPY supplies market context. No model is trained yet."),
    ("features", "Create clues and answers", "Clues use information available after each close. The answer is whether buying at the next open and selling 5 trading sessions later earns a positive return, including dividends. Future answers are for training and evaluation; they are never model inputs."),
    ("splits", "Separate past from future", "The model learns from older dates and is evaluated on later dates. Each fold is one evaluation window. fit_cutoff is the last date whose completed outcomes may be used for training. The latest 20% of eligible dates forms the final test period."),
    ("train", "Learn from older examples", "Train separate models for each time window using AAPL only. XGBoost learns patterns in the clues; logistic regression and simple baselines provide comparisons. Training uses only outcomes already known at that window's cutoff."),
    ("predict", "Inspect predictions", "probability is the model's estimate of a positive return. A probability at or above 0.55 creates a buy signal. target is the actual historical answer: 1 means positive, 0 means zero or negative. A prediction can be wrong."),
    ("backtest", "Simulate trades", "A buy signal enters at the following open, if no AAPL position is already held. Exit is 5 trading sessions after entry. Cash is held between trades. Slippage makes buys more expensive and sells cheaper; 10 basis points means 0.10% per side."),
    ("report", "Review the experiment", "Compare prediction accuracy with trading returns and buying and holding. Accuracy alone does not tell you whether trading was profitable. The saved report records this experiment's results."),
]

# These explanations are shown before execution, so a learner knows what the
# button will do even when there is no output yet.
GUIDES = {
    "collect": ("The run's dates, AAPL symbol and SPY benchmark.",
        "Build the NYSE session calendar; download daily bars (or generate demo bars); reject invalid prices and duplicate dates; report missing sessions; check whether each instrument has enough history.",
        "Open is the first price of a session; close is the last. High/low describe the day's range. Volume counts traded units. Check coverage and quality before continuing."),
    "features": ("The saved daily price and volume history.",
        "For every session, calculate trailing returns, moving averages, RSI, MACD, volatility, volume and candle clues. Add SPY context. Use the next open as entry and the open 5 sessions after entry as exit to create the historical answer. Discard incomplete feature windows from training.",
        "One row is one historical decision. Clue columns look backward; entry_date, exit_date, forward_return and target describe its later outcome. The first months need warmup, and the last rows cannot yet have a future answer."),
    "splits": ("Dates with complete model inputs.",
        "Reserve the latest 20% of dates for final testing. Start with the configured initial training history, then evaluate in successive windows. Before each window, expand training using only labels whose exit date is at or before fit_cutoff.",
        "Development windows precede final-test windows. All dates in a window are later than its training cutoff. A 5-session outcome must finish before training can use it."),
    "train": ("Past feature rows and known 0/1 answers for each window.",
        "Fit XGBoost's sequence of decision trees and logistic regression's weighted input model. Fit always-up and majority baselines too. Save each window's models so later predictions can be reproduced. Model settings are fixed, with no automated tuning.",
        "The table below shows how many examples each window actually learns from. More training rows become available in later windows. These are trained models, not measured prediction results yet."),
    "predict": ("Each saved model and the later rows in its evaluation window.",
        "Ask the model for the probability of target=1, attach the historical outcome for scoring, and mark a trading signal when the probability meets the configured threshold. Also calculate momentum and baseline comparisons.",
        "Probability is an estimate, not a guarantee. Classification accuracy uses 50%; buying uses the configured threshold. Compare the selected prediction with its actual outcome below."),
    "backtest": ("Saved signals, opening prices, dividends and starting capital.",
        "Walk through sessions in order. Credit eligible dividends, process scheduled exits, then buy when a qualifying signal arrives and AAPL is not already held. Apply slippage to both buys and sells, and record daily cash and holdings. Skip signals that cannot exit within the evaluation period.",
        "A signal does not always become a trade: the portfolio may already hold AAPL. Net return is the total percentage gain/loss; drawdown measures a fall from a previous portfolio peak. Compare against buy-and-hold over the same dates."),
    "report": ("The saved accuracy tables, portfolio results and run configuration.",
        "Assemble a standalone HTML report from this experiment's saved results. This step does not retrain models or place orders.",
        "Review the final-test results, compare baselines and trading costs, and keep the report with the experiment. Synthetic results only demonstrate the workflow."),
}


def example_outcome(run, rows, key):
    """Connect one persisted decision to its source prices without retraining."""
    valid = rows.dropna(subset=["target", "entry_date", "exit_date"]).sort_values("date")
    if valid.empty:
        st.info("No completed outcomes are available for this selection.")
        return
    choice = st.selectbox("Inspect a decision date", range(len(valid)),
        format_func=lambda i: str(valid.iloc[i].date.date()), key=f"example_{key}_{run.name}")
    row = valid.iloc[choice]
    bars = pd.read_parquet(run / "bars.parquet")
    apple = bars[bars.ticker.eq("AAPL")].set_index("date")
    entry = apple.loc[row.entry_date, "open"]
    exit_price = apple.loc[row.exit_date, "open"]
    distributions = apple.loc[(apple.index > row.entry_date) & (apple.index <= row.exit_date), "dividend"].sum()
    st.write(f"Decision after close on {row.date.date()}. Enter at the open on {row.entry_date.date()} (${entry:,.4f}); exit at the open on {row.exit_date.date()} (${exit_price:,.4f}).")
    st.code(f"Return = (exit price + dividends per unit) / entry price - 1\n       = ({exit_price:.4f} + {distributions:.4f}) / {entry:.4f} - 1\n       = {row.forward_return:.2%}\nTarget = {int(row.target)} ({'positive' if row.target else 'zero or negative'} return)", language="text")
    st.caption("This target is before trading slippage. Displayed prices are rounded; calculations use full precision.")
    pass  # English explanations are provided above.
    if "probability" in valid.columns:
        threshold = manifest(run)["config"]["threshold"]
        st.write(f"Model probability: {row.probability:.1%}. Buy threshold: {threshold:.1%}. Signal: {'buy candidate' if row.probability >= threshold else 'no new purchase'}. Actual direction: {'up' if row.target else 'zero or down'}.")
        decision = "" if row.probability >= threshold else ""
        pass  # English explanations are provided above.


def show_setup(run=None):
    st.subheader("Setup: understand your experiment")
    show_flowchart("setup")
    st.write("Start with one stock, AAPL, and one question: will its return be positive over 5 trading sessions, starting at the next open?")
    st.markdown("**1. Prepare Python and start the dashboard**")
    st.code('python -m venv .venv\n.\\.venv\\Scripts\\python -m pip install -e ".[dev]"\n.\\.venv\\Scripts\\python -m streamlit run streamlit_app.py', language="powershell")
    st.caption("Run these commands from the project folder. Installation needs internet access. If this dashboard is already open, setup is working.")
    st.markdown("**2. Choose your data in the sidebar**")
    st.write("Keep Learn with one stock (AAPL) checked. Synthetic demo creates repeatable, invented prices for offline practice. Uncheck Synthetic demo to download actual historical AAPL and SPY prices from Yahoo Finance. All 50 instruments applies to ordinary research runs; learning mode always uses AAPL only.")
    st.markdown("**3. Create an experiment**")
    st.write("Click Create run. This saves the settings in a new experiment folder; it does not download prices or train a model. Each later lesson has its own Run button. SPY provides market clues and a benchmark; it is not another training stock in this mode.")
    if run is None:
        st.info("Create a run using the sidebar to see its exact settings and begin step 1.")
        return
    m = manifest(run)
    cfg = m["config"]
    settings = [
        ("Data source", "Generated synthetic prices" if m["synthetic"] else "Yahoo Finance historical prices", "Demo prices do not represent Apple's historical performance."),
        ("Training stock", ", ".join(cfg["tickers"]), "Only these stocks supply training examples and strategy positions."),
        ("Dates", f"{cfg['start']} to {cfg['end']} (end excluded)", "Only exchange sessions in this interval are requested."),
        ("Prediction horizon", str(cfg["horizons"]), "Trading sessions between entry open and exit open."),
        ("Minimum history", cfg["minimum_sessions"], "An instrument needs this many valid bars to qualify."),
        ("Initial training history", cfg["initial_train_sessions"], "Feature-eligible session dates before the first evaluation window; warmup occurs earlier."),
        ("Final test fraction", f"{cfg['test_fraction']:.0%}", "Latest feature-eligible dates, reserved for final evaluation."),
        ("Window size", cfg["fold_sessions"], "Retrain before each window using only outcomes known by its cutoff."),
        ("Starting capital", f"${cfg['capital']:,.0f}", "Simulated cash; this app does not place live orders."),
        ("Buy threshold", f"{cfg['threshold']:.0%}", "Minimum predicted probability required for a buy candidate."),
        ("Slippage cases", str([0] + cfg["slippage_bps"]), "Basis points charged per side; 10 bps = 0.10%."),
        ("Random seed", cfg["seed"], "Controls generated prices and model randomness for repeatability."),
    ]
    st.markdown("**4. Review the exact settings saved for this run**")
    st.dataframe(pd.DataFrame(settings, columns=["Setting", "Value", "Meaning"]).astype(str), hide_index=True)
    st.caption("Edit config/research.json before creating a new run to change research settings. Demo mode overrides dates, minimum history, window sizes and tree count. Existing runs keep their saved settings.")
    st.markdown("**5. Follow the process**")
    st.write("Download prices → build inputs and historical answers → define time windows → train → predict → simulate trades → review results. Run one step, inspect it, then continue. Changing a lesson does not run the pipeline.")
    st.write(f"This experiment's outputs are saved in {run}. The manifest records settings and stage completion; later steps save tables, models and the report there.")


def explain_training(run):
    """Show actual training rows and the exact declared learning procedure."""
    from .modeling import fold_masks
    cfg = manifest(run)["config"]
    schema = read_json(run / "feature_schema.json")
    folds = read_json(run / "splits.json")["folds"]
    data = pd.read_parquet(run / "features.parquet").dropna(subset=schema["columns"])
    st.markdown("**Training, operation by operation**")
    choice = st.selectbox("Training window to inspect", range(len(folds)),
        format_func=lambda i: f"Window {folds[i]['fold']}: {folds[i]['start']} to {folds[i]['end']}", key=f"fit_{run.name}")
    fold = folds[choice]
    mask, _ = fold_masks(data, fold)
    fit = data[mask]
    st.write(f"1. Select rows with complete input columns, decision date at or before {fold['fit_cutoff']}, a known target, and exit date at or before that cutoff. This window uses {len(fit):,} AAPL examples.")
    st.write(f"2. Build X, the input table: {len(fit):,} rows × {len(schema['columns'])} columns. Build y, the answer column: {int(fit.target.sum()):,} positive outcomes and {int(fit.target.eq(0).sum()):,} zero/negative outcomes. Dates, ticker, future returns, entry/exit dates and targets are excluded from X.")
    with st.expander("Inspect the actual X input columns and sample rows"):
        st.write(schema["columns"])
        st.dataframe(fit[["date"] + schema["columns"]].head(10), hide_index=True)
    st.write("Matching y answers for those same rows:")
    st.dataframe(fit[["date", "entry_date", "exit_date", "forward_return", "target"]].head(10), hide_index=True)
    st.write("3. Fit XGBoost. Start with an initial estimate, then add decision trees in sequence. Each new tree uses the current prediction errors to improve the binary log-loss objective. The result is an estimated probability of target=1. No hand-written indicator rule determines the answer.")
    meanings = {"n_estimators": "Number of boosting rounds (trees).", "max_depth": "Maximum depth of a tree; limits its complexity.", "learning_rate": "Scales each new tree's contribution.", "subsample": "Fraction of training rows sampled per tree.", "colsample_bytree": "Fraction of input columns sampled per tree."}
    st.dataframe(pd.DataFrame([{"Parameter": k, "Value": str(v), "Meaning": meanings.get(k, "Additional configured XGBoost setting.")} for k, v in cfg["xgboost"].items()]), hide_index=True)
    st.write("4. Fit comparisons on the same rows. Logistic regression first standardizes each input using training-only means and scales, then fits weighted inputs (C=0.1, max_iter=500). Always-up predicts 1; majority predicts the most common training answer. XGBoost uses the inputs without that standardization. Momentum uses a positive trailing 5-session return and needs no fitting.")
    st.code('model.fit(X_train, y_train)\n# Later, on rows in this window only:\nprobability_up = model.predict_proba(X_evaluation)[:, 1]', language="python")
    st.caption("Illustration of the learning API; the application also handles class ordering and saves metadata. This inspector does not fit models or change results.")
    st.write(f"5. Save this window's models, input column order, cutoff and training metadata in models/h5_f{fold['fold']}.joblib. Repeat for the next window with more matured historical examples. Settings stay fixed; there is no automated tuning or early stopping.")
    st.write("6. Evaluate in the next lesson. Training itself is not evidence of accuracy: predictions must be checked on the later rows. During the final test, scheduled retraining can use earlier test outcomes only after their exits have occurred.")


def show_learning(run):
    m = manifest(run)
    st.subheader("Learn with Apple, one step at a time")
    if not m.get("learning"):
        st.info("Create a new run with 'Learn with one stock (AAPL)' checked to use this walkthrough.")
        return
    st.write("One stock: AAPL · One horizon: 5 trading sessions. Each button runs just one stage. Stop and inspect its output before continuing.")
    st.caption("Uncheck Synthetic demo when creating a run to download historical prices. Synthetic prices are generated examples, not Apple's actual history.")
    completed = m["stages"]
    next_index = next((i for i, (stage, _, _) in enumerate(LESSONS) if stage not in completed), len(LESSONS))
    st.progress(next_index / len(LESSONS), text=f"{next_index} of {len(LESSONS)} steps complete")
    options = [-1] + list(range(len(LESSONS)))
    initial = 0 if next_index == 0 else min(next_index, len(LESSONS)-1)+1
    index = st.selectbox("Lesson", options, index=initial,
                         format_func=lambda i: "Setup: start here" if i == -1 else f"{i+1}. {LESSONS[i][1]}", key=f"lesson_{run.name}")
    if index == -1:
        show_setup(run)
        def start():
            st.session_state[f"lesson_{run.name}"] = 0
        st.button("Continue to step 1: Download prices", on_click=start)
        return
    stage, title, explanation = LESSONS[index]
    st.subheader(f"{index+1}. {title}")
    with st.expander("Flowchart with example", expanded=True):
        show_flowchart(stage)
    st.write(explanation)
    inputs, operation, interpretation = GUIDES[stage]
    st.markdown("**What goes into this step**")
    st.write(inputs)
    st.markdown("**What happens when you run it**")
    st.write(operation)
    st.markdown("**How to understand the output**")
    st.write(interpretation)
    if stage == "train" and "splits" in completed:
        explain_training(run)
    if index > next_index:
        st.info(f"Complete step {next_index+1} first.")
        return
    already_complete = stage in completed
    label = f"Run again: {title}" if already_complete else f"Run step {index+1}: {title}"
    if already_complete:
        st.success("This step is complete. Its saved results are below.")
        st.caption("Running it again clears completion of later steps. Those steps must then be run again to refresh their results.")
    if st.button(label, type="secondary" if already_complete else "primary"):
        from .__main__ import execute
        try:
            with st.spinner(f"{title}..."):
                execute(stage, run)
            st.success("Step complete. Inspect the output below, then select the next lesson when ready.")
        except Exception as exc:
            st.error(str(exc))
            return
    elif not already_complete:
        return
    st.caption("Saved output from this step")
    if stage == "collect":
        bars = pd.read_parquet(run / "bars.parquet")
        apple = bars[bars.ticker.eq("AAPL")].sort_values("date")
        st.line_chart(apple, x="date", y="close")
        st.dataframe(apple.tail(20), hide_index=True)
        st.dataframe(pd.read_csv(run / "coverage.csv"), hide_index=True)
        st.json(read_json(run / "quality.json"))
    elif stage == "features":
        data = pd.read_parquet(run / "features.parquet")
        cols = ["date", "close"] if "close" in data else ["date"]
        cols += ["return_5", "sma_distance_20", "rsi", "relative_volume", "entry_date", "exit_date", "forward_return", "target"]
        valid = data.dropna(subset=["return_5", "sma_distance_20", "target"])
        st.dataframe(valid[cols].tail(20), hide_index=True)
        st.caption("return_5 = trailing 5-session return; sma_distance_20 = distance from the 20-session average; RSI = recent upward/downward price strength; relative_volume = volume divided by its 20-session average. Returns are decimals: 0.02 means 2%.")
        example_outcome(run, valid, "features")
    elif stage == "splits":
        st.dataframe(pd.DataFrame(read_json(run / "splits.json")["folds"]), hide_index=True)
    elif stage == "train":
        from .modeling import fold_masks
        schema = read_json(run / "feature_schema.json")
        data = pd.read_parquet(run / "features.parquet").dropna(subset=schema["columns"])
        summaries = []
        for fold in read_json(run / "splits.json")["folds"]:
            mask, _ = fold_masks(data, fold)
            fit = data[mask]
            summaries.append({"window": fold["fold"], "period": fold["phase"],
                "training_examples": len(fit), "positive_answers": int(fit.target.sum()),
                "last_training_date": fit.date.max(), "last_known_exit": fit.exit_date.max(),
                "evaluation_start": fold["start"]})
        st.dataframe(pd.DataFrame(summaries), hide_index=True)
        st.write("Settings used for XGBoost:")
        st.json(m["config"]["xgboost"])
        st.write("Models saved locally for each window:")
        st.write([p.name for p in sorted((run / "models").glob("*.joblib"))])
        st.caption("Each file contains XGBoost, logistic regression, always-up and majority models for one window. The momentum comparison is added during prediction.")
    elif stage == "predict":
        predictions = pd.read_parquet(run / "predictions.parquet")
        selected = predictions[predictions.model.eq("xgboost") & predictions.phase.eq("test")]
        st.dataframe(selected[["date", "entry_date", "exit_date", "probability", "signal", "target", "forward_return"]].tail(20), hide_index=True)
        st.dataframe(pd.read_csv(run / "accuracy.csv"), hide_index=True)
        st.caption("Accuracy uses a 0.50 classification threshold; simulated trades use 0.55. Blank targets at the end have no known future outcome.")
        example_outcome(run, selected, "predict")
    elif stage == "backtest":
        performance = pd.read_csv(run / "performance.csv")
        selected = performance[performance.phase.eq("test") & performance.bps.eq(10)]
        st.dataframe(selected, hide_index=True)
        result = selected[selected.model.eq("xgboost")].iloc[0]
        curve = pd.read_parquet(run / "backtests" / f"{result.artifact}_equity.parquet")
        st.line_chart(curve, x="date", y="equity")
        st.write("Actual simulated buys and sells:")
        trades = pd.read_parquet(run / "backtests" / f"{result.artifact}_trades.parquet")
        st.dataframe(trades.head(20), hide_index=True)
        if not trades.empty:
            choice = st.selectbox("Inspect a trade", range(len(trades)),
                format_func=lambda i: f"{trades.iloc[i].date.date()}: {trades.iloc[i].side}", key=f"trade_{run.name}")
            trade = trades.iloc[choice]
            st.write(f"{trade.side.title()} {trade.normalized_units:,.4f} normalized units. Opening reference price: ${trade.reference_price:,.4f}; executed price after slippage: ${trade.executed_price:,.4f}. Transaction value: ${trade.notional:,.2f}. Slippage cost: ${trade.slippage_cost:,.2f}.")
            action = "" if trade.side == "buy" else ""
            pass  # English explanations are provided above.
            if pd.notna(trade.realized_pnl):
                st.write(f"This completed position earned/lost ${trade.realized_pnl:,.2f}, including distributions and execution costs, over {int(trade.holding_sessions)} trading sessions.")
                pass  # English explanations are provided above.
        else:
            st.info("No trades met the execution rules in this period.")
        st.caption("equal_weight is buy-and-hold AAPL in this one-stock run. SPY is the market comparison. Returns are decimals: 0.10 means 10%.")
    else:
        scores = pd.read_csv(run / "accuracy.csv")
        st.write("Final-test prediction results:")
        st.dataframe(scores[scores.phase.eq("test")][["model", "samples", "accuracy", "precision", "recall"]], hide_index=True)
        st.caption("Accuracy = fraction of correct directions. Precision = fraction of predicted up moves that were positive. Recall = fraction of actual positive moves identified. These use the 50% classification threshold.")
        performance = pd.read_csv(run / "performance.csv")
        st.write("Final-test portfolio results at 10 basis points per side:")
        st.dataframe(performance[performance.phase.eq("test") & performance.bps.eq(10)][["model", "net_return", "cagr", "sharpe", "max_drawdown"]], hide_index=True)
        st.caption("CAGR = annualized growth; Sharpe = return relative to day-to-day variability; max_drawdown = largest decline from a prior peak. Returns are decimals: 0.10 means 10%.")
        st.download_button("Download this experiment's report", (run / "report.html").read_bytes(), file_name="aapl_learning_report.html", mime="text/html")
    if index < len(LESSONS)-1:
        def advance():
            st.session_state[f"lesson_{run.name}"] = index+1
        st.button(f"Continue to step {index+2}: {LESSONS[index+1][1]}", on_click=advance)
        st.info(f"When ready, select lesson {index+2}: {LESSONS[index+1][1]} above.")
