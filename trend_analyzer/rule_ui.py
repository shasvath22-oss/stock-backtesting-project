from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from .common import digest, read_json, write_json
from .data import sessions
from .swing_backtest import rule_signals, simulate_rule
from .periods import period_windows, period_settings, period_caption


def show_rule_backtest(run, bars, manifest, analysis, selected_period=None):
    title = "Step 4A — Old-period backtest" if selected_period == "old" else "Step 4B — New-period backtest" if selected_period == "new" else "Step 4 — Trading backtest"
    st.subheader(title)
    st.caption("Run calculates and saves both periods once. Select Step 4A or Step 4B on the left to view each period's results.")
    research_periods = manifest.get("research_periods",period_settings())
    st.caption(period_caption(research_periods))
    st.write("")
    with st.expander("Execution rules", expanded=True):
        st.write("Fixed long-only HH → HL rule. Entry open must lie strictly between stop and target; otherwise skip. Protective orders are active on the entry candle too. Stops/targets execute on wick touches (unlike the strict break counts in Step 3). If both are touched in one daily bar, stop wins. A gap below stop exits at the open; a gap above target fills at target conservatively. Slippage is then applied adversely to both fills, including target fills.")
        st.write("Count the entry day as holding session 1; if no stop or target, exit at the close of session N. Liquidate at the final available close if history ends earlier. No new trade while holding, including a session when an existing trade exits. Dividends accrue only after the entry date. Quantities use the vendor's split-normalized basis; do not apply splits twice.")
        st.write("10 bps per side baseline; identical confirmation-time signals at 5/10/20 bps. Buy-and-hold uses the same price interval, capital and costs. All settings and entry/exit audits are saved. This uses historical data already explored in statistics; it is not an untouched test or proof of future profitability.")
    with st.form(f"rule_{analysis.name}"):
        capital = st.number_input("Initial capital", min_value=100., value=10000., step=100.)
        hold = st.number_input("Maximum holding sessions", min_value=1,max_value=252,value=20)
        submit = st.form_submit_button("Run step 4: Backtest HH → HL",type="primary")
    root = analysis / "rule_backtests"
    if submit:
        calendar = sessions(manifest["start"],pd.Timestamp(manifest["end_exclusive"])-pd.Timedelta(days=1))
        windows = period_windows(calendar,research_periods)
        required_dates = windows["old"].union(windows["new"])
        if len(required_dates.difference(pd.DatetimeIndex(bars.date))):
            st.error("Price gaps must be resolved before execution simulation.")
            return
        events = pd.read_parquet(analysis / "events.parquet")
        signals = rule_signals(bars,events)
        results, frames = [],{}
        with st.spinner("Simulating identical signals at 5, 10 and 20 bps..."):
            for period, dates in windows.items():
                if len(dates) < 2:
                    continue
                period_bars = bars[bars.date.between(dates[0],dates[-1])]
                period_signals = signals[signals.confirmed_at.between(dates[0],dates[-1])]
                for cost in [5,10,20]:
                    for model, benchmark in [("HH_HL",False),("buy_hold",True)]:
                        metrics,curve,trades,audit = simulate_rule(period_bars,period_signals,float(capital),cost,int(hold),benchmark)
                        results.append(metrics | {"model":model,"period":period})
                        prefix = f"{period}_{model}_{cost}bps"
                        for name,frame in [("equity",curve),("trades",trades),("audit",audit)]:
                            frames[f"{prefix}_{name}.parquet"] = frame
        if not results:
            st.error("No sessions in the selected research periods. Collect prices covering the dates selected on the left.")
            return
        output = root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output.mkdir(parents=True)
        signals.to_parquet(output / "signals.parquet",index=False)
        for name,frame in frames.items():
            frame.to_parquet(output / name,index=False)
        pd.DataFrame(results).to_csv(output / "performance.csv",index=False)
        write_json(output / "settings.json", {"rule":"long_HH_HL_v1","capital":capital,"max_hold":hold,
            "baseline_bps":10,"cost_cases":[5,10,20],"price_sha256":digest(run / "bars.parquet"),
            "events_sha256":digest(analysis / "events.parquet"),"signals_sha256":digest(output / "signals.parquet"),
            "research_periods":research_periods,"version":"period_backtest_v2"})
        st.session_state[f"rule_run_{analysis.name}"] = str(output)
        st.success("Backtest saved")
    saved = sorted([p for p in root.glob("*") if (p / "settings.json").exists()],reverse=True)
    if not saved:
        return
    output = Path(st.selectbox("Saved rule backtest",[str(p) for p in saved],key=f"rule_run_{analysis.name}"))
    settings = read_json(output / "settings.json")
    if "research_periods" in settings:
        st.caption("Saved results: " + period_caption(settings["research_periods"]))
        if settings["research_periods"] != research_periods:
            st.info("These results use earlier period settings. Run Step 4 to use the dates selected on the left.")
    if settings["price_sha256"] != digest(run / "bars.parquet") or settings["events_sha256"] != digest(analysis / "events.parquet"):
        st.error("Inputs changed; rerun the backtest.")
        return
    st.caption(f"Saved run: capital ${settings['capital']:,.0f}, maximum {settings['max_hold']} sessions. Form changes apply only on Run.")
    performance = pd.read_csv(output / "performance.csv")
    if "period" not in performance.columns:
        st.info("This saved backtest used the full history. Run step 4 again for the configured old/new periods.")
        return
    st.caption("Each period starts with fresh capital; positions close at that period's last close. No trades or observation windows cross from old into new. Earlier confirmed swings may supply context, but new-period signals must be confirmed in the new period. Old/new is a date split, not a claim that previously viewed data is untouched.")
    cfg = settings["research_periods"]
    old_end = cfg.get("old_end") or str((pd.Timestamp(cfg["new_start"])-pd.Timedelta(days=1)).date())
    periods = [selected_period] if selected_period else ["old","new"]
    for period in periods:
        title = f"Step 4A: {cfg['old_start']} to {old_end}" if period == "old" else f"Step 4B: {cfg['new_start']} to {cfg.get('new_end') or 'latest saved session'}"
        st.subheader(title)
        subset = performance[performance.period.eq(period)]
        if subset.empty:
            st.info("No saved data for this period. Collect prices covering these dates, rerun Step 2 and Step 4.")
        else:
            show_period_results(output,settings,subset,period)


def show_period_results(output,settings,performance,period):
    cost = st.selectbox("Execution cost per side",[5,10,20],index=1,key=f"rule_cost_{output.name}_{period}")
    chosen = performance[performance.model.eq("HH_HL") & performance.bps.eq(cost)].iloc[0]
    gain = chosen.final_equity-settings["capital"]
    trades = pd.read_parquet(output / f"{period}_HH_HL_{cost}bps_trades.parquet")
    total_profit = trades.loc[trades.pnl.gt(0),"pnl"].sum()
    total_loss = trades.loc[trades.pnl.lt(0),"pnl"].sum()
    profit, loss, net = st.columns(3)
    profit.metric("Total profit", f"${total_profit:,.2f}")
    loss.metric("Total loss", f"${total_loss:,.2f}")
    net.metric("Net profit / loss ($)", f"${gain:+,.2f}")
    st.caption("Total profit = sum of profitable trades; total loss = sum of losing trades (negative). Net = profit + loss. Slippage and eligible dividends are already included; do not deduct costs again.")
    balance, returns, count = st.columns(3)
    balance.metric("Final capital", f"${chosen.final_equity:,.2f}")
    returns.metric("Net return", f"{chosen.net_return:.2%}")
    count.metric("Completed trades", int(chosen.trades))
    st.dataframe(performance,hide_index=True)
    st.caption("net_return=0.10 means 10%; negative drawdown is a decline from a peak. win_rate uses completed trades. HH_HL = strategy; buy_hold = hold this ETF/stock throughout. No signals may mean zero trades.")
    curves = []
    for model in ["HH_HL","buy_hold"]:
        curve = pd.read_parquet(output / f"{period}_{model}_{cost}bps_equity.parquet")
        curves.append(curve.set_index("date").equity.rename(model))
    st.line_chart(pd.concat(curves,axis=1),y_label="Portfolio value")
    st.dataframe(trades,hide_index=True)
    if not trades.empty:
        index = st.selectbox("Explain one trade",list(range(len(trades))),
            format_func=lambda i: str(trades.iloc[i].entry_date.date()),key=f"rule_trade_{output.name}_{period}_{cost}")
        trade = trades.iloc[index]
        st.write(f"Signal {trade.confirmed_at.date()} → buy {trade.entry_date.date()} at ${trade.entry_price:.4f} ({trade.units:.4f} units); stop ${trade.stop:.4f}, target ${trade.target:.4f}. Sell {trade.exit_date.date()} at ${trade.exit_price:.4f}; reason: {trade.reason}. Dividends ${trade.dividends:.2f}; profit/loss ${trade.pnl:+.2f}.")
        pass  # English explanations are provided above.
    with st.expander("Signals and skipped entries"):
        signals = pd.read_parquet(output / "signals.parquet")
        dates = period_windows(pd.DatetimeIndex(signals.confirmed_at),settings["research_periods"])[period]
        st.dataframe(signals[signals.confirmed_at.isin(dates)],hide_index=True)
        st.dataframe(pd.read_parquet(output / f"{period}_HH_HL_{cost}bps_audit.parquet"),hide_index=True)
    st.download_button("Download trade CSV",trades.to_csv(index=False),file_name=f"{period}_HH_HL_{cost}bps_trades.csv",mime="text/csv",key=f"trade_download_{output.name}_{period}")
