"""Dependency-free SVG learning diagrams with a worked, hypothetical example."""
from html import escape

import streamlit as st


STEPS = [
    ("setup", "Setup", "AAPL · 5 trading sessions · starting capital"),
    ("collect", "1. Prices", "Download AAPL + SPY; check missing data"),
    ("features", "2. Inputs + answers", "Past indicators → X; later outcome → y"),
    ("splits", "3. Time split", "Older dates → training; later dates → evaluation"),
    ("train", "4. Train", "Fit models using only completed past outcomes"),
    ("predict", "5. Predict", "Estimate probability of a positive return"),
    ("backtest", "6. Simulate", "Signal + execution rules → trades + costs"),
    ("report", "7. Review", "Accuracy + profit/loss + comparison"),
]


def diagram(rows, active=None):
    """Render SVG with escaped labels; no Graphviz binary or package required."""
    height = len(rows) * 110 + 10
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="620" height="{height}" viewBox="0 0 620 {height}" role="img">',
        '<title>Learning pipeline flowchart</title>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#475569"/></marker></defs>']
    for i, (key, title, detail) in enumerate(rows):
        y = i * 110 + 5
        fill = "#dbeafe" if key == active else "#f1f5f9"
        stroke = "#2563eb" if key == active else "#64748b"
        parts.append(f'<rect x="10" y="{y}" width="600" height="82" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        for offset, text, size in [(30, title, 18), (59, detail, 15)]:
            parts.append(f'<text x="310" y="{y+offset}" text-anchor="middle" fill="#0f172a" font-family="Nirmala UI, Latha, Arial, sans-serif" font-size="{size}">{escape(text)}</text>')
        if i < len(rows)-1:
            parts.append(f'<line x1="310" y1="{y+84}" x2="310" y2="{y+104}" stroke="#475569" stroke-width="2" marker-end="url(#arrow)"/>')
    parts.append('</svg>')
    return "".join(parts)


def show_flowchart(active="setup"):
    st.subheader("Flowchart")
    st.caption("Follow the arrows from top to bottom. Blue marks this lesson.")
    overview, example = st.columns(2)
    with overview:
        st.markdown("**Full process")
        st.image(diagram(STEPS, active), width="stretch")
    with example:
        st.markdown("**Apple example")
        scenario = st.selectbox("Example outcome", ["Buy → profit", "Buy → loss", "No buy"], key=f"flow_example_{active}")
        buy = not scenario.startswith("No buy")
        exit_price = 95 if "loss" in scenario else 105
        rows = [
            ("setup", "Hypothetical AAPL example", "$1,000 capital · 5 sessions · no dividends"),
            ("collect", "Observe older prices", "Calculate clues after the market closes"),
            ("train", "Previously trained", "Learn from older X rows and known y answers"),
            ("predict", "Probability", "62% ≥ 55% → buy candidate" if buy else "48% < 55% → no new purchase"),
        ]
        if buy:
            rows += [
                ("backtest", "Next open", "$100 per unit → 10 units before costs"),
                ("backtest", "Exit 5 sessions later", f"Sell at ${exit_price} → ${10*exit_price:,} before costs"),
                ("report", "Result", f"${10*(exit_price-100):+} before costs; target = {int(exit_price > 100)}"),
            ]
        else:
            rows += [("backtest", "Stay in cash", "Assume no existing position; cash stays $1,000"),
                     ("report", "No trade", "No gain/loss from this skipped candidate")]
        st.image(diagram(rows, active), width="stretch")
        st.caption("Invented numbers, not saved run results. Actual trades also depend on whether AAPL is already held and whether an exit is available.")
        if buy:
            units = 1000 / (100 * 1.001)
            proceeds = units * exit_price * .999
            st.write(f"With 10 bps slippage per side: buy {units:.4f} normalized units at $100.10; sell at ${exit_price*.999:.3f}. Final cash ≈ ${proceeds:,.2f}; profit/loss ≈ ${proceeds-1000:+,.2f}.")
            pass  # English explanations are provided above.
    st.info("Training reads only past clues and completed answers. At prediction time, the later answer is hidden; it is used afterward for evaluation.")
