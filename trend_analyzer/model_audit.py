"""Post-prediction audit only: never used to fit or select the model."""
import numpy as np
import pandas as pd

from .swing_backtest import simulate_rule


def audit_candidates(bars,candidates,predictions,model_trades,baseline_trades,model_entries,
                     capital=10000,bps=10,max_hold=20):
    rows = []
    data = bars.sort_values("date").reset_index(drop=True)
    for signal in candidates.itertuples(index=False):
        prediction = predictions[predictions.confirmed_at.eq(signal.confirmed_at)]
        probability = float(prediction.iloc[0].probability) if len(prediction) else np.nan
        selected = bool(prediction.iloc[0].selected) if len(prediction) else False
        decision = "selected" if selected else "rejected" if len(prediction) else "missing_inputs"
        one = pd.DataFrame([signal._asdict()])
        _,_,trades,entries = simulate_rule(data,one,capital,bps,max_hold)
        record = {"confirmed_at":signal.confirmed_at,"probability":probability,"decision":decision,
            "stop":signal.stop,"target":signal.target,"hypothetical_pnl":np.nan,"hypothetical_return":np.nan,
            "entry_date":pd.NaT,"exit_date":pd.NaT,"exit_reason":None,"outcome":None,
            "actual_model_pnl":np.nan,"actual_baseline_pnl":np.nan,"model_execution":decision}
        if trades.empty:
            record["outcome"] = entries.iloc[0].status if len(entries) else "no_executable_next_session"
        else:
            trade = trades.iloc[0]
            record.update(entry_date=trade.entry_date,exit_date=trade.exit_date,exit_reason=trade.reason)
            if trade.reason == "end_of_data_close":
                record["outcome"] = "incomplete_history"
            else:
                record.update(hypothetical_pnl=trade.pnl,hypothetical_return=trade.pnl/capital,
                    outcome="profit" if trade.pnl>0 else "loss" if trade.pnl<0 else "flat")
        for actual,column in [(model_trades,"actual_model_pnl"),(baseline_trades,"actual_baseline_pnl")]:
            match = actual[actual.confirmed_at.eq(signal.confirmed_at)]
            if len(match):
                record[column] = match.iloc[0].pnl
        if selected:
            if pd.notna(record["actual_model_pnl"]):
                record["model_execution"] = "executed"
            else:
                future = data[data.date.gt(signal.confirmed_at)]
                matches = model_entries[model_entries.entry_date.eq(future.iloc[0].date)] if len(future) else model_entries.iloc[:0]
                record["model_execution"] = matches.iloc[0].status if len(matches) else "no_executable_next_session"
        rows.append(record)
    result = pd.DataFrame(rows,columns=["confirmed_at","probability","decision","stop","target","hypothetical_pnl",
        "hypothetical_return","entry_date","exit_date","exit_reason","outcome","actual_model_pnl","actual_baseline_pnl","model_execution"])
    for column in ["confirmed_at","entry_date","exit_date"]:
        result[column] = pd.to_datetime(result[column])
    return result
