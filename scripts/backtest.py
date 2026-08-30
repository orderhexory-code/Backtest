#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

IMPULSE_LOOKBACK = 3
PULLBACK_MAX_BARS = 3

MIN_IMPULSE_BODY = 0.55
MAX_PULLBACK_RETRACE = 0.60

RISK_REWARD = 2.0

INITIAL_BALANCE = 10_000.0
RISK_PER_TRADE = 100.0

ONE_TRADE_AT_A_TIME = True


# ============================================================
# LOAD DATA
# ============================================================

def load_data(csv_file):

    df = pd.read_csv(csv_file)

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df[required].copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna()

    df = df.sort_values(
        "timestamp"
    )

    df = df.drop_duplicates(
        subset="timestamp"
    )

    df = df.reset_index(drop=True)

    return df


# ============================================================
# CANDLE INFORMATION
# ============================================================

def candle_body(row):

    return abs(
        row["close"] - row["open"]
    )


def candle_range(row):

    return (
        row["high"] - row["low"]
    )


def bullish(row):

    return row["close"] > row["open"]


def bearish(row):

    return row["close"] < row["open"]


# ============================================================
# PULLBACK BACKTEST
# ============================================================

def run_backtest(df):

    trades = []

    balance = INITIAL_BALANCE

    equity_curve = []

    i = IMPULSE_LOOKBACK

    while i < len(df) - PULLBACK_MAX_BARS - 2:

        row = df.iloc[i]

        current_range = candle_range(row)

        if current_range <= 0:

            i += 1
            continue

        body_ratio = (
            candle_body(row)
            / current_range
        )

        # ----------------------------------------------------
        # LONG IMPULSE
        # ----------------------------------------------------

        if bullish(row) and body_ratio >= MIN_IMPULSE_BODY:

            impulse_high = row["high"]
            impulse_low = row["low"]

            impulse_range = (
                impulse_high - impulse_low
            )

            j = i + 1

            pullback_low = float("inf")
            pullback_bars = 0

            while (
                j < len(df)
                and pullback_bars < PULLBACK_MAX_BARS
            ):

                pb = df.iloc[j]

                pullback_bars += 1

                pullback_low = min(
                    pullback_low,
                    pb["low"]
                )

                # Structure invalidated
                if pb["low"] < impulse_low:

                    break

                retracement = (
                    impulse_high - pb["low"]
                ) / impulse_range

                # Pullback too deep
                if retracement > MAX_PULLBACK_RETRACE:

                    break

                # --------------------------------------------
                # Bullish reversal candle
                # --------------------------------------------

                if bullish(pb):

                    entry = pb["high"]

                    stop = pullback_low

                    risk = entry - stop

                    if risk <= 0:

                        break

                    target = (
                        entry
                        + risk * RISK_REWARD
                    )

                    # ----------------------------------------
                    # Find trade result
                    # ----------------------------------------

                    exit_price = None
                    exit_time = None
                    result = None

                    k = j + 1

                    while k < len(df):

                        future = df.iloc[k]

                        # Both SL and TP hit
                        # on same candle:
                        # conservative assumption = SL first
                        if (
                            future["low"] <= stop
                            and future["high"] >= target
                        ):

                            exit_price = stop
                            result = "LOSS"

                            exit_time = future[
                                "timestamp"
                            ]

                            break

                        if future["low"] <= stop:

                            exit_price = stop
                            result = "LOSS"

                            exit_time = future[
                                "timestamp"
                            ]

                            break

                        if future["high"] >= target:

                            exit_price = target
                            result = "WIN"

                            exit_time = future[
                                "timestamp"
                            ]

                            break

                        k += 1

                    # Trade still open at end
                    if exit_price is None:

                        future = df.iloc[-1]

                        exit_price = future["close"]

                        exit_time = future[
                            "timestamp"
                        ]

                        result = "OPEN"

                    if result == "WIN":

                        r_multiple = RISK_REWARD
                        pnl = RISK_PER_TRADE * RISK_REWARD

                    elif result == "LOSS":

                        r_multiple = -1
                        pnl = -RISK_PER_TRADE

                    else:

                        r_multiple = (
                            exit_price - entry
                        ) / risk

                        pnl = (
                            r_multiple
                            * RISK_PER_TRADE
                        )

                    balance += pnl

                    trades.append({

                        "side": "LONG",

                        "entry_time": pb[
                            "timestamp"
                        ],

                        "entry": entry,

                        "stop": stop,

                        "target": target,

                        "exit_time": exit_time,

                        "exit": exit_price,

                        "result": result,

                        "r": r_multiple,

                        "pnl": pnl,

                        "balance": balance,

                    })

                    i = k

                    break

                j += 1

        # ----------------------------------------------------
        # SHORT IMPULSE
        # ----------------------------------------------------

        elif bearish(row) and body_ratio >= MIN_IMPULSE_BODY:

            impulse_high = row["high"]
            impulse_low = row["low"]

            impulse_range = (
                impulse_high - impulse_low
            )

            j = i + 1

            pullback_high = -float("inf")

            pullback_bars = 0

            while (
                j < len(df)
                and pullback_bars < PULLBACK_MAX_BARS
            ):

                pb = df.iloc[j]

                pullback_bars += 1

                pullback_high = max(
                    pullback_high,
                    pb["high"]
                )

                # Structure invalidated
                if pb["high"] > impulse_high:

                    break

                retracement = (
                    pb["high"] - impulse_low
                ) / impulse_range

                if retracement > MAX_PULLBACK_RETRACE:

                    break

                # --------------------------------------------
                # Bearish reversal candle
                # --------------------------------------------

                if bearish(pb):

                    entry = pb["low"]

                    stop = pullback_high

                    risk = stop - entry

                    if risk <= 0:

                        break

                    target = (
                        entry
                        - risk * RISK_REWARD
                    )

                    exit_price = None
                    exit_time = None
                    result = None

                    k = j + 1

                    while k < len(df):

                        future = df.iloc[k]

                        # Both hit:
                        # conservative assumption = SL first
                        if (
                            future["high"] >= stop
                            and future["low"] <= target
                        ):

                            exit_price = stop
                            result = "LOSS"

                            exit_time = future[
                                "timestamp"
                            ]

                            break

                        if future["high"] >= stop:

                            exit_price = stop
                            result = "LOSS"

                            exit_time = future[
                                "timestamp"
                            ]

                            break

                        if future["low"] <= target:

                            exit_price = target
                            result = "WIN"

                            exit_time = future[
                                "timestamp"
                            ]

                            break

                        k += 1

                    if exit_price is None:

                        future = df.iloc[-1]

                        exit_price = future["close"]

                        exit_time = future[
                            "timestamp"
                        ]

                        result = "OPEN"

                    if result == "WIN":

                        r_multiple = RISK_REWARD
                        pnl = RISK_PER_TRADE * RISK_REWARD

                    elif result == "LOSS":

                        r_multiple = -1
                        pnl = -RISK_PER_TRADE

                    else:

                        r_multiple = (
                            entry - exit_price
                        ) / risk

                        pnl = (
                            r_multiple
                            * RISK_PER_TRADE
                        )

                    balance += pnl

                    trades.append({

                        "side": "SHORT",

                        "entry_time": pb[
                            "timestamp"
                        ],

                        "entry": entry,

                        "stop": stop,

                        "target": target,

                        "exit_time": exit_time,

                        "exit": exit_price,

                        "result": result,

                        "r": r_multiple,

                        "pnl": pnl,

                        "balance": balance,

                    })

                    i = k

                    break

                j += 1

        i += 1

    return pd.DataFrame(trades)


# ============================================================
# STATISTICS
# ============================================================

def calculate_stats(trades):

    if trades.empty:

        return {

            "trades": 0,

            "wins": 0,

            "losses": 0,

            "win_rate": 0,

            "net_pnl": 0,

            "profit_factor": 0,

            "max_drawdown": 0,

            "return_pct": 0,

        }

    closed = trades[
        trades["result"].isin(
            ["WIN", "LOSS"]
        )
    ].copy()

    wins = (
        closed["pnl"] > 0
    ).sum()

    losses = (
        closed["pnl"] < 0
    ).sum()

    gross_profit = closed.loc[
        closed["pnl"] > 0,
        "pnl"
    ].sum()

    gross_loss = abs(
        closed.loc[
            closed["pnl"] < 0,
            "pnl"
        ].sum()
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = 0

    equity = (
        pd.Series(
            [INITIAL_BALANCE]
            + trades["balance"].tolist()
        )
    )

    peak = equity.cummax()

    drawdown = (
        equity - peak
    )

    max_drawdown = abs(
        drawdown.min()
    )

    final_balance = (
        trades["balance"].iloc[-1]
        if not trades.empty
        else INITIAL_BALANCE
    )

    return {

        "trades": len(closed),

        "wins": int(wins),

        "losses": int(losses),

        "win_rate": (
            wins / len(closed) * 100
            if len(closed)
            else 0
        ),

        "net_pnl": (
            final_balance
            - INITIAL_BALANCE
        ),

        "profit_factor": profit_factor,

        "max_drawdown": max_drawdown,

        "return_pct": (
            (
                final_balance
                / INITIAL_BALANCE
            ) - 1
        ) * 100,

    }


# ============================================================
# HTML DASHBOARD
# ============================================================

def create_html(df, trades, stats, output):

    candles = []

    for _, row in df.iterrows():

        candles.append({

            "time": int(
                row["timestamp"].timestamp()
            ),

            "open": float(row["open"]),

            "high": float(row["high"]),

            "low": float(row["low"]),

            "close": float(row["close"]),

        })


    trade_data = []

    for _, row in trades.iterrows():

        trade_data.append({

            "side": row["side"],

            "entryTime": int(
                row["entry_time"].timestamp()
            ),

            "exitTime": int(
                row["exit_time"].timestamp()
            ),

            "entry": float(row["entry"]),

            "exit": float(row["exit"]),

            "stop": float(row["stop"]),

            "target": float(row["target"]),

            "result": row["result"],

            "pnl": float(row["pnl"]),

            "r": float(row["r"]),

        })


    candles_json = json.dumps(
        candles
    )

    trades_json = json.dumps(
        trade_data
    )

    html = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>US100 M1 Pullback Backtest</title>

<script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>

<style>

body {{
    margin: 0;
    background: #0b0f14;
    color: #e6edf3;
    font-family: Arial, sans-serif;
}}

.container {{
    padding: 20px;
}}

h1 {{
    margin-bottom: 20px;
}}

.stats {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
}}

.card {{
    background: #111821;
    border: 1px solid #263241;
    border-radius: 10px;
    padding: 15px;
}}

.label {{
    color: #8b98a9;
    font-size: 13px;
}}

.value {{
    font-size: 23px;
    margin-top: 6px;
    font-weight: bold;
}}

.chart {{
    margin-top: 20px;
    background: #111821;
    border: 1px solid #263241;
    border-radius: 10px;
    overflow: hidden;
}}

#chart {{
    height: 600px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 20px;
}}

th, td {{
    padding: 9px;
    border-bottom: 1px solid #263241;
    text-align: left;
}}

th {{
    color: #8b98a9;
}}

</style>

</head>

<body>

<div class="container">

<h1>US100 M1 Pullback Backtest</h1>

<div class="stats">

<div class="card">
<div class="label">Trades</div>
<div class="value">{stats["trades"]}</div>
</div>

<div class="card">
<div class="label">Win Rate</div>
<div class="value">{stats["win_rate"]:.2f}%</div>
</div>

<div class="card">
<div class="label">Net P&L</div>
<div class="value">${stats["net_pnl"]:.2f}</div>
</div>

<div class="card">
<div class="label">Profit Factor</div>
<div class="value">{stats["profit_factor"]:.2f}</div>
</div>

<div class="card">
<div class="label">Max Drawdown</div>
<div class="value">${stats["max_drawdown"]:.2f}</div>
</div>

<div class="card">
<div class="label">Return</div>
<div class="value">{stats["return_pct"]:.2f}%</div>
</div>

</div>

<div class="chart">

<div id="chart"></div>

</div>

<h2>Trades</h2>

<table>

<thead>

<tr>
<th>Side</th>
<th>Entry</th>
<th>Exit</th>
<th>Entry Price</th>
<th>Exit Price</th>
<th>Result</th>
<th>R</th>
<th>P&L</th>
</tr>

</thead>

<tbody id="trades"></tbody>

</table>

</div>

<script>

const candles = {candles_json};

const trades = {trades_json};

const chart = LightweightCharts.createChart(
    document.getElementById("chart"),
    {{
        layout: {{
            background: {{ color: "#111821" }},
            textColor: "#d1d9e0"
        }},
        grid: {{
            vertLines: {{ color: "#1d2733" }},
            horzLines: {{ color: "#1d2733" }}
        }},
        timeScale: {{
            timeVisible: true,
            secondsVisible: false
        }}
    }}
);

const series = chart.addCandlestickSeries();

series.setData(candles);

for (const trade of trades) {{

    const color =
        trade.side === "LONG"
            ? "#22c55e"
            : "#ef4444";

    series.setMarkers([
        {{
            time: trade.entryTime,
            position:
                trade.side === "LONG"
                    ? "belowBar"
                    : "aboveBar",
            color: color,
            shape:
                trade.side === "LONG"
                    ? "arrowUp"
                    : "arrowDown",
            text:
                trade.side + " " +
                trade.result
        }}
    ]);

}}

chart.timeScale().fitContent();


const tbody =
    document.getElementById("trades");


for (const trade of trades) {{

    const tr =
        document.createElement("tr");

    tr.innerHTML = `
        <td>${{trade.side}}</td>
        <td>${{new Date(trade.entryTime * 1000).toISOString()}}</td>
        <td>${{new Date(trade.exitTime * 1000).toISOString()}}</td>
        <td>${{trade.entry.toFixed(3)}}</td>
        <td>${{trade.exit.toFixed(3)}}</td>
        <td>${{trade.result}}</td>
        <td>${{trade.r.toFixed(2)}}R</td>
        <td>$${{trade.pnl.toFixed(2)}}</td>
    `;

    tbody.appendChild(tr);

}}

</script>

</body>

</html>
"""

    Path(output).write_text(
        html,
        encoding="utf-8"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv",
        help="Input OHLC CSV"
    )

    parser.add_argument(
        "--output",
        default="backtest.html",
        help="Output HTML file"
    )

    args = parser.parse_args()

    print("Loading CSV...")

    df = load_data(args.csv)

    print(
        f"Loaded {len(df):,} candles."
    )

    print("Running pullback backtest...")

    trades = run_backtest(df)

    stats = calculate_stats(trades)

    print()
    print("===================================")
    print("BACKTEST RESULTS")
    print("===================================")

    for key, value in stats.items():

        print(
            f"{key}: {value}"
        )

    create_html(
        df,
        trades,
        stats,
        args.output
    )

    print()
    print(
        f"HTML dashboard created: {args.output}"
    )


if __name__ == "__main__":
    main()
