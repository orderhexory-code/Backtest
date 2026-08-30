#!/usr/bin/env python3
"""
US100 M1 Price-Action Scalping Backtester
Deterministic M1 Opening Range Breakout + Retest Scalper (Zero Look-Ahead Bias)
"""

import argparse
import datetime
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Configuration Data Structures
# ----------------------------------------------------------------------
class BacktestConfig:
    def __init__(
        self,
        csv_path: str,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 100.0,
        risk_reward_ratio: float = 2.0,
        spread_points: float = 1.0,
        slippage_points: float = 0.5,
        commission_per_trade: float = 0.0,
        or_candles_count: int = 5,
        max_retest_candles: int = 3,
        audit_mode: bool = True,
        output_dir: str = "results",
        html_output: str = "backtest.html",
    ):
        self.csv_path = csv_path
        self.initial_balance = float(initial_balance)
        self.risk_per_trade = float(risk_per_trade)
        self.risk_reward_ratio = float(risk_reward_ratio)
        self.spread_points = float(spread_points)
        self.slippage_points = float(slippage_points)
        self.commission_per_trade = float(commission_per_trade)
        self.or_candles_count = int(or_candles_count)
        self.max_retest_candles = int(max_retest_candles)
        self.audit_mode = bool(audit_mode)
        self.output_dir = output_dir
        self.html_output = html_output


# ----------------------------------------------------------------------
# 1. Data Ingestion & Validation
# ----------------------------------------------------------------------
def load_and_validate_data(csv_path: str) -> pd.DataFrame:
    """
    Loads M1 CSV, validates schema, converts timestamp to UTC, sorts chronologically,
    removes duplicates and malformed rows.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {e}")

    required_columns = ["timestamp", "open", "high", "low", "close"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    initial_len = len(df)
    if initial_len == 0:
        raise ValueError("Input CSV is empty.")

    # Validate and cast numeric columns
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Reject malformed OHLC values (NaNs, negatives, inverted high/low)
    df = df.dropna(subset=required_columns).copy()
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    df = df[(df["high"] >= df["low"]) & (df["high"] >= df["open"]) & (df["high"] >= df["close"]) &
            (df["low"] <= df["open"]) & (df["low"] <= df["close"])]

    # Timestamp conversion (Unix ms or s or ISO)
    try:
        # Check if numeric
        df["timestamp_num"] = pd.to_numeric(df["timestamp"], errors="coerce")
        if df["timestamp_num"].notnull().all():
            # If timestamp > 1e11 it's in milliseconds
            if df["timestamp_num"].iloc[0] > 1e11:
                df["datetime"] = pd.to_datetime(df["timestamp_num"], unit="ms", utc=True)
            else:
                df["datetime"] = pd.to_datetime(df["timestamp_num"], unit="s", utc=True)
        else:
            df["datetime"] = pd.to_datetime(df["timestamp"], utc=True)
    except Exception as e:
        raise ValueError(f"Failed to convert timestamps: {e}")

    df = df.dropna(subset=["datetime"]).copy()
    
    # Sort chronologically and drop duplicate timestamps
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    final_len = len(df)
    print(f"[DATA] Loaded {final_len} valid M1 candles from {csv_path} (Dropped {initial_len - final_len} malformed/duplicate rows).")
    print(f"[DATA] Range: {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")
    
    return df


# ----------------------------------------------------------------------
# 2. Strict Event-Driven Backtesting Engine
# ----------------------------------------------------------------------
class BacktestEngine:
    def __init__(self, df: pd.DataFrame, config: BacktestConfig):
        self.df = df
        self.cfg = config
        self.trades: List[Dict[str, Any]] = []
        self.audit_log: List[Dict[str, Any]] = []

    def run(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
        n = len(self.df)
        if n < self.cfg.or_candles_count + 2:
            print("[WARN] Not enough candles to execute strategy.")
            stats = calculate_statistics(self.trades, self.cfg, self.df)
            return self.trades, stats, self.audit_log

        # Extract numpy arrays for fast sequential bar iteration
        opens = self.df["open"].to_numpy()
        highs = self.df["high"].to_numpy()
        lows = self.df["low"].to_numpy()
        closes = self.df["close"].to_numpy()
        datetimes = self.df["datetime"].to_numpy()

        # Group candles by session / calendar day
        # Extract dates in UTC
        dates = pd.to_datetime(datetimes).date

        # Session tracking state
        current_session_date = None
        session_bar_idx = 0
        or_high = None
        or_low = None
        or_established = False

        # Strategy setup tracking state
        # Setup states: 'SEARCHING_BREAKOUT', 'WAITING_RETEST', 'READY_ENTRY'
        setup_state = "IDLE"
        breakout_side = None  # 'LONG' or 'SHORT'
        breakout_bar_idx = None
        retest_candles_elapsed = 0
        pending_signal = None  # Will hold signal triggered at bar i for execution at bar i+1

        # Position tracking state
        open_trade: Optional[Dict[str, Any]] = None
        trade_counter = 0

        for i in range(n):
            c_date = dates[i]
            c_open = opens[i]
            c_high = highs[i]
            c_low = lows[i]
            c_close = closes[i]
            c_dt = str(datetimes[i])

            # Detect new day/session boundary
            if c_date != current_session_date:
                current_session_date = c_date
                session_bar_idx = 0
                or_high = None
                or_low = None
                or_established = False
                if open_trade is None:
                    setup_state = "SEARCHING_BREAKOUT"
                    pending_signal = None
            else:
                session_bar_idx += 1

            # ----------------------------------------------------------
            # STEP A: Execute Pending Next-Bar Orders (if any) at Bar Open
            # ----------------------------------------------------------
            if pending_signal is not None and open_trade is None:
                # Next candle OPEN execution
                side = pending_signal["side"]
                sig_sl = pending_signal["sl_price"]
                sig_bar = pending_signal["signal_bar_index"]

                if side == "LONG":
                    entry_price = c_open + self.cfg.slippage_points + self.cfg.spread_points
                    risk_points = entry_price - sig_sl
                    if risk_points <= 0:
                        # Invalid SL logic guard
                        risk_points = 1.0
                    reward_points = risk_points * self.cfg.risk_reward_ratio
                    target_price = entry_price + reward_points
                else:  # SHORT
                    entry_price = c_open - self.cfg.slippage_points
                    risk_points = sig_sl - entry_price
                    if risk_points <= 0:
                        risk_points = 1.0
                    reward_points = risk_points * self.cfg.risk_reward_ratio
                    target_price = entry_price - reward_points

                trade_counter += 1
                open_trade = {
                    "trade_id": trade_counter,
                    "date": str(c_date),
                    "side": side,
                    "signal_time": pending_signal["signal_time"],
                    "entry_time": c_dt,
                    "exit_time": None,
                    "entry_price": round(entry_price, 4),
                    "stop_price": round(sig_sl, 4),
                    "target_price": round(target_price, 4),
                    "exit_price": None,
                    "risk_points": round(risk_points, 4),
                    "reward_points": round(reward_points, 4),
                    "R_multiple": 0.0,
                    "gross_pnl": 0.0,
                    "cost": round(self.cfg.commission_per_trade, 4),
                    "net_pnl": 0.0,
                    "result": "OPEN",
                    "exit_reason": "NONE",
                    "signal_bar_index": sig_bar,
                    "entry_bar_index": i,
                    "exit_bar_index": None,
                }
                pending_signal = None
                setup_state = "TRADE_ACTIVE"

            # ----------------------------------------------------------
            # STEP B: Manage Active Open Trade against Bar i High / Low
            # ----------------------------------------------------------
            if open_trade is not None:
                side = open_trade["side"]
                ep = open_trade["entry_price"]
                sl = open_trade["stop_price"]
                tp = open_trade["target_price"]
                risk_pts = open_trade["risk_points"]

                trade_closed = False
                exit_price = 0.0
                exit_reason = ""
                result = ""

                # Conservative Rule: Check Same-Candle SL/TP Ambiguity
                if side == "LONG":
                    hit_sl = c_low <= sl
                    hit_tp = c_high >= tp

                    if hit_sl and hit_tp:
                        # Ambiguous: SL FIRST conservative rule
                        trade_closed = True
                        exit_price = sl - self.cfg.slippage_points
                        exit_reason = "AMBIGUOUS_SL_TP_SL_FIRST"
                        result = "LOSS"
                    elif hit_sl:
                        trade_closed = True
                        exit_price = sl - self.cfg.slippage_points
                        exit_reason = "STOP_LOSS"
                        result = "LOSS"
                    elif hit_tp:
                        trade_closed = True
                        exit_price = tp
                        exit_reason = "TAKE_PROFIT"
                        result = "WIN"
                else:  # SHORT
                    hit_sl = c_high >= sl
                    hit_tp = c_low <= tp

                    if hit_sl and hit_tp:
                        trade_closed = True
                        exit_price = sl + self.cfg.slippage_points + self.cfg.spread_points
                        exit_reason = "AMBIGUOUS_SL_TP_SL_FIRST"
                        result = "LOSS"
                    elif hit_sl:
                        trade_closed = True
                        exit_price = sl + self.cfg.slippage_points + self.cfg.spread_points
                        exit_reason = "STOP_LOSS"
                        result = "LOSS"
                    elif hit_tp:
                        trade_closed = True
                        exit_price = tp
                        exit_reason = "TAKE_PROFIT"
                        result = "WIN"

                # Check End-of-Data condition
                if not trade_closed and i == n - 1:
                    trade_closed = True
                    exit_price = c_close
                    exit_reason = "END_OF_DATA"
                    if side == "LONG":
                        result = "WIN" if exit_price > ep else ("LOSS" if exit_price < ep else "BREAKEVEN")
                    else:
                        result = "WIN" if exit_price < ep else ("LOSS" if exit_price > ep else "BREAKEVEN")

                if trade_closed:
                    open_trade["exit_time"] = c_dt
                    open_trade["exit_price"] = round(exit_price, 4)
                    open_trade["exit_bar_index"] = i
                    open_trade["exit_reason"] = exit_reason
                    open_trade["result"] = result

                    # PnL & R calculation
                    if side == "LONG":
                        price_diff = exit_price - ep
                    else:
                        price_diff = ep - exit_price

                    # Fixed dollar risk model: 1R = risk_per_trade
                    r_multiple = price_diff / risk_pts if risk_pts > 0 else 0.0
                    open_trade["R_multiple"] = round(r_multiple, 4)

                    gross_pnl = r_multiple * self.cfg.risk_per_trade
                    net_pnl = gross_pnl - open_trade["cost"]
                    open_trade["gross_pnl"] = round(gross_pnl, 2)
                    open_trade["net_pnl"] = round(net_pnl, 2)

                    self.trades.append(open_trade)
                    open_trade = None
                    setup_state = "SEARCHING_BREAKOUT"

                # Skip signal searching while trade is active
                continue

            # ----------------------------------------------------------
            # STEP C: Opening Range Construction
            # ----------------------------------------------------------
            if session_bar_idx < self.cfg.or_candles_count:
                # Still building opening range
                continue
            elif session_bar_idx == self.cfg.or_candles_count:
                # Exactly candle #5 closes (0..4) -> compute immutable OR
                # Only use historical slice up to i (which is exactly the 5th candle)
                or_slice_highs = highs[i - self.cfg.or_candles_count + 1 : i + 1]
                or_slice_lows = lows[i - self.cfg.or_candles_count + 1 : i + 1]
                or_high = float(np.max(or_slice_highs))
                or_low = float(np.min(or_slice_lows))
                or_established = True
                setup_state = "SEARCHING_BREAKOUT"
                continue

            if not or_established or or_high is None or or_low is None:
                continue

            # ----------------------------------------------------------
            # STEP D: Signal Scan & State Machine (Strict Zero Look-Ahead)
            # ----------------------------------------------------------
            if setup_state == "SEARCHING_BREAKOUT":
                # Check Long Breakout: Close above OR_HIGH
                if c_close > or_high:
                    setup_state = "WAITING_RETEST"
                    breakout_side = "LONG"
                    breakout_bar_idx = i
                    retest_candles_elapsed = 0
                # Check Short Breakout: Close below OR_LOW
                elif c_close < or_low:
                    setup_state = "WAITING_RETEST"
                    breakout_side = "SHORT"
                    breakout_bar_idx = i
                    retest_candles_elapsed = 0

            elif setup_state == "WAITING_RETEST":
                retest_candles_elapsed += 1
                if retest_candles_elapsed > self.cfg.max_retest_candles:
                    # Retest window expired without rejection
                    setup_state = "SEARCHING_BREAKOUT"
                    breakout_side = None
                    breakout_bar_idx = None
                    continue

                if breakout_side == "LONG":
                    # Invalidation: candle breaks below OR_LOW
                    if c_low < or_low:
                        setup_state = "SEARCHING_BREAKOUT"
                        continue

                    # Retest condition: trades near/into OR_HIGH (low <= or_high)
                    # Bullish rejection: closes bullish (close > open) and closes above OR_HIGH
                    is_retest = c_low <= or_high * 1.0005  # Within 0.05% or touching OR High
                    is_bullish_rejection = (c_close > c_open) and (c_close >= or_high)

                    if is_retest and is_bullish_rejection:
                        # Structure low for SL: minimum low from breakout bar to signal bar
                        struct_lows = lows[breakout_bar_idx : i + 1]
                        sl_price = float(np.min(struct_lows))

                        pending_signal = {
                            "side": "LONG",
                            "signal_time": c_dt,
                            "signal_bar_index": i,
                            "sl_price": sl_price,
                        }

                        if self.cfg.audit_mode:
                            self.audit_log.append({
                                "signal_bar_index": i,
                                "signal_time": c_dt,
                                "available_data_until": c_dt,
                                "signal_reason": f"Bullish Rejection at OR_HIGH {or_high:.2f}",
                                "expected_entry_bar": i + 1,
                                "side": "LONG",
                                "stop_loss": sl_price,
                            })
                        setup_state = "READY_ENTRY"

                elif breakout_side == "SHORT":
                    # Invalidation: candle breaks above OR_HIGH
                    if c_high > or_high:
                        setup_state = "SEARCHING_BREAKOUT"
                        continue

                    # Retest condition: trades near/into OR_LOW (high >= or_low)
                    # Bearish rejection: closes bearish (close < open) and closes below OR_LOW
                    is_retest = c_high >= or_low * 0.9995
                    is_bearish_rejection = (c_close < c_open) and (c_close <= or_low)

                    if is_retest and is_bearish_rejection:
                        # Structure high for SL: maximum high from breakout bar to signal bar
                        struct_highs = highs[breakout_bar_idx : i + 1]
                        sl_price = float(np.max(struct_highs))

                        pending_signal = {
                            "side": "SHORT",
                            "signal_time": c_dt,
                            "signal_bar_index": i,
                            "sl_price": sl_price,
                        }

                        if self.cfg.audit_mode:
                            self.audit_log.append({
                                "signal_bar_index": i,
                                "signal_time": c_dt,
                                "available_data_until": c_dt,
                                "signal_reason": f"Bearish Rejection at OR_LOW {or_low:.2f}",
                                "expected_entry_bar": i + 1,
                                "side": "SHORT",
                                "stop_loss": sl_price,
                            })
                        setup_state = "READY_ENTRY"

        stats = calculate_statistics(self.trades, self.cfg, self.df)
        return self.trades, stats, self.audit_log


# ----------------------------------------------------------------------
# 3. Analytics & Statistics Computation
# ----------------------------------------------------------------------
def calculate_statistics(trades: List[Dict[str, Any]], cfg: BacktestConfig, df: pd.DataFrame) -> Dict[str, Any]:
    total_trades = len(trades)
    initial_balance = cfg.initial_balance

    if total_trades == 0:
        return {
            "initial_balance": initial_balance,
            "final_balance": initial_balance,
            "gross_pnl": 0.0,
            "total_costs": 0.0,
            "net_pnl": 0.0,
            "return_pct": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "breakeven_trades": 0,
            "win_rate": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "average_r": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown_dollars": 0.0,
            "max_drawdown_pct": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "long_win_rate": 0.0,
            "short_win_rate": 0.0,
            "average_holding_time_mins": 0.0,
            "best_trade_id": None,
            "worst_trade_id": None,
            "candle_count": len(df),
            "date_range": [str(df['datetime'].iloc[0]), str(df['datetime'].iloc[-1])] if len(df) > 0 else ["", ""],
        }

    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] < 0]
    bes = [t for t in trades if t["net_pnl"] == 0]

    gross_pnl = sum(t["gross_pnl"] for t in trades)
    total_costs = sum(t["cost"] for t in trades)
    net_pnl = sum(t["net_pnl"] for t in trades)
    final_balance = initial_balance + net_pnl
    return_pct = (net_pnl / initial_balance) * 100.0

    win_rate = (len(wins) / total_trades) * 100.0 if total_trades > 0 else 0.0
    avg_win = float(np.mean([t["net_pnl"] for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([abs(t["net_pnl"]) for t in losses])) if losses else 0.0
    avg_r = float(np.mean([t["R_multiple"] for t in trades])) if trades else 0.0

    sum_win_dollars = sum(t["net_pnl"] for t in wins)
    sum_loss_dollars = sum(abs(t["net_pnl"]) for t in losses)
    profit_factor = (sum_win_dollars / sum_loss_dollars) if sum_loss_dollars > 0 else (999.0 if sum_win_dollars > 0 else 0.0)

    # Expectancy ($ per trade)
    expectancy = (win_rate / 100.0 * avg_win) - ((1.0 - (win_rate / 100.0)) * avg_loss)

    # Equity Curve & Drawdown calculation
    equity = initial_balance
    equity_curve = [initial_balance]
    peak = initial_balance
    max_dd_dollars = 0.0
    max_dd_pct = 0.0

    for t in trades:
        equity += t["net_pnl"]
        equity_curve.append(equity)
        if equity > peak:
            peak = equity
        dd_dollars = peak - equity
        dd_pct = (dd_dollars / peak) * 100.0 if peak > 0 else 0.0
        if dd_dollars > max_dd_dollars:
            max_dd_dollars = dd_dollars
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    largest_win = max([t["net_pnl"] for t in trades]) if trades else 0.0
    largest_loss = min([t["net_pnl"] for t in trades]) if trades else 0.0

    best_trade = max(trades, key=lambda x: x["net_pnl"]) if trades else None
    worst_trade = min(trades, key=lambda x: x["net_pnl"]) if trades else None

    # Side breakdown
    longs = [t for t in trades if t["side"] == "LONG"]
    shorts = [t for t in trades if t["side"] == "SHORT"]
    long_wins = [t for t in longs if t["net_pnl"] > 0]
    short_wins = [t for t in shorts if t["net_pnl"] > 0]

    long_win_rate = (len(long_wins) / len(longs) * 100.0) if longs else 0.0
    short_win_rate = (len(short_wins) / len(shorts) * 100.0) if shorts else 0.0

    # Holding times
    holding_bars = [t["exit_bar_index"] - t["entry_bar_index"] for t in trades if t["exit_bar_index"] is not None]
    avg_holding_bars = float(np.mean(holding_bars)) if holding_bars else 0.0

    return {
        "initial_balance": round(initial_balance, 2),
        "final_balance": round(final_balance, 2),
        "gross_pnl": round(gross_pnl, 2),
        "total_costs": round(total_costs, 2),
        "net_pnl": round(net_pnl, 2),
        "return_pct": round(return_pct, 2),
        "total_trades": total_trades,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "breakeven_trades": len(bes),
        "win_rate": round(win_rate, 2),
        "average_win": round(avg_win, 2),
        "average_loss": round(avg_loss, 2),
        "average_r": round(avg_r, 2),
        "profit_factor": round(profit_factor, 2),
        "expectancy": round(expectancy, 2),
        "max_drawdown_dollars": round(max_dd_dollars, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "largest_win": round(largest_win, 2),
        "largest_loss": round(largest_loss, 2),
        "long_trades": len(longs),
        "short_trades": len(shorts),
        "long_win_rate": round(long_win_rate, 2),
        "short_win_rate": round(short_win_rate, 2),
        "average_holding_time_mins": round(avg_holding_bars, 1),
        "best_trade_id": best_trade["trade_id"] if best_trade else None,
        "worst_trade_id": worst_trade["trade_id"] if worst_trade else None,
        "candle_count": len(df),
        "date_range": [str(df["datetime"].iloc[0]), str(df["datetime"].iloc[-1])],
    }


# ----------------------------------------------------------------------
# 4. HTML Dashboard Generator
# ----------------------------------------------------------------------
def generate_html_dashboard(
    trades: List[Dict[str, Any]],
    stats: Dict[str, Any],
    df: pd.DataFrame,
    cfg: BacktestConfig,
    output_path: str
):
    """
    Generates a professional, self-contained, interactive dark analytics dashboard.
    """
    # Downsample candles for interactive chart if too large, maintaining start/end bounds
    sample_df = df.copy()
    if len(sample_df) > 3000:
        # Take the most recent 3000 bars or trade regions for rendering responsiveness
        sample_df = sample_df.tail(3000)

    candle_data_json = json.dumps([
        {
            "time": str(row["datetime"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        for _, row in sample_df.iterrows()
    ])

    trades_json = json.dumps(trades)
    stats_json = json.dumps(stats)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>US100 M1 Scalping Backtest — Professional Analytics</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root {{
            --bg-primary: #0a0e17;
            --bg-secondary: #121826;
            --bg-card: #1a2234;
            --border-color: #26334d;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --accent-amber: #f59e0b;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }}
        body {{
            background-color: var(--bg-primary);
            color: var(--text-primary);
            padding: 24px;
            line-height: 1.5;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 24px;
        }}
        .header h1 {{
            font-size: 24px;
            font-weight: 700;
            letter-spacing: -0.5px;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .badge {{
            font-size: 12px;
            padding: 4px 8px;
            border-radius: 4px;
            background: #1e293b;
            color: var(--accent-blue);
            border: 1px solid var(--border-color);
            font-weight: 600;
        }}
        .meta-info {{
            font-size: 13px;
            color: var(--text-secondary);
            text-align: right;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            transition: transform 0.15s ease;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
        }}
        .kpi-title {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 22px;
            font-weight: 700;
        }}
        .text-green {{ color: var(--accent-green); }}
        .text-red {{ color: var(--accent-red); }}
        .text-blue {{ color: var(--accent-blue); }}
        .text-amber {{ color: var(--accent-amber); }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 24px;
        }}
        @media (max-width: 1024px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .chart-box {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
        }}
        .chart-box.full-width {{
            grid-column: 1 / -1;
        }}
        .chart-title {{
            font-size: 15px;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .table-container {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 24px;
            overflow-x: auto;
        }}
        .table-controls {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 16px;
            gap: 12px;
        }}
        input, select {{
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 13px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
        }}
        th {{
            background-color: var(--bg-secondary);
            padding: 10px 12px;
            color: var(--text-secondary);
            font-weight: 600;
            border-bottom: 1px solid var(--border-color);
            cursor: pointer;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-color);
        }}
        tr:hover {{
            background-color: rgba(255, 255, 255, 0.02);
        }}
        .tag {{
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}
        .tag-win {{ background: rgba(16, 185, 129, 0.2); color: var(--accent-green); }}
        .tag-loss {{ background: rgba(239, 68, 68, 0.2); color: var(--accent-red); }}
        .tag-long {{ background: rgba(59, 130, 246, 0.2); color: var(--accent-blue); }}
        .tag-short {{ background: rgba(245, 158, 11, 0.2); color: var(--accent-amber); }}
    </style>
</head>
<body>

    <div class="header">
        <div>
            <h1>US100 M1 SCALPING BACKTEST <span class="badge">v1.0 Pure Price Action</span></h1>
            <div style="font-size: 13px; color: var(--text-secondary); margin-top: 4px;">
                Strategy: M1 Opening Range Breakout + Retest Scalper | Zero Indicators | Zero Look-Ahead Bias
            </div>
        </div>
        <div class="meta-info">
            <div><strong>Dataset:</strong> {os.path.basename(cfg.csv_path)} ({stats['candle_count']:,} candles)</div>
            <div><strong>Period:</strong> {stats['date_range'][0]} to {stats['date_range'][1]}</div>
            <div><strong>Costs:</strong> Spread {cfg.spread_points} pt | Slip {cfg.slippage_points} pt | Comm ${cfg.commission_per_trade}</div>
        </div>
    </div>

    <!-- KPI Cards -->
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Net P&L</div>
            <div class="kpi-value {'text-green' if stats['net_pnl'] >= 0 else 'text-red'}">
                ${stats['net_pnl']:,.2f}
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Return %</div>
            <div class="kpi-value {'text-green' if stats['return_pct'] >= 0 else 'text-red'}">
                {stats['return_pct']:.2f}%
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Win Rate</div>
            <div class="kpi-value text-blue">
                {stats['win_rate']:.1f}%
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Profit Factor</div>
            <div class="kpi-value {'text-green' if stats['profit_factor'] >= 1.0 else 'text-red'}">
                {stats['profit_factor']:.2f}
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Max Drawdown</div>
            <div class="kpi-value text-red">
                {stats['max_drawdown_pct']:.2f}% (${stats['max_drawdown_dollars']:,.2f})
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Total Trades</div>
            <div class="kpi-value text-primary">
                {stats['total_trades']}
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Average R</div>
            <div class="kpi-value {'text-green' if stats['average_r'] >= 0 else 'text-red'}">
                {stats['average_r']:+.2f}R
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Total Costs</div>
            <div class="kpi-value text-amber">
                ${stats['total_costs']:,.2f}
            </div>
        </div>
    </div>

    <!-- Charts Grid -->
    <div class="charts-grid">
        <!-- Equity Curve Chart -->
        <div class="chart-box full-width">
            <div class="chart-title">
                <span>Equity Curve & Drawdown Profile</span>
                <span style="font-size:12px; color:var(--text-secondary)">Starting Balance: ${stats['initial_balance']:,.2f}</span>
            </div>
            <div style="height: 320px;">
                <canvas id="equityChart"></canvas>
            </div>
        </div>

        <!-- Trade P&L Distribution -->
        <div class="chart-box">
            <div class="chart-title">Trade-by-Trade R-Multiple Performance</div>
            <div style="height: 260px;">
                <canvas id="pnlBarChart"></canvas>
            </div>
        </div>

        <!-- Win/Loss Pie & Breakdown -->
        <div class="chart-box">
            <div class="chart-title">Outcome Distribution</div>
            <div style="height: 260px;">
                <canvas id="winLossChart"></canvas>
            </div>
        </div>

        <!-- Long vs Short Breakdown -->
        <div class="chart-box">
            <div class="chart-title">Long vs Short Performance</div>
            <div style="height: 260px;">
                <canvas id="sideChart"></canvas>
            </div>
        </div>

        <!-- Exit Reason Distribution -->
        <div class="chart-box">
            <div class="chart-title">Exit Reason Breakdown (Audit)</div>
            <div style="height: 260px;">
                <canvas id="exitReasonChart"></canvas>
            </div>
        </div>

        <!-- Hour of Day Performance -->
        <div class="chart-box full-width">
            <div class="chart-title">Hour of Day (UTC) P&L Distribution</div>
            <div style="height: 260px;">
                <canvas id="hourlyChart"></canvas>
            </div>
        </div>
    </div>

    <!-- Trade Ledger Table -->
    <div class="table-container">
        <div class="chart-title">
            <span>Complete Trade Ledger</span>
            <div class="table-controls">
                <input type="text" id="tableSearch" placeholder="Search trades (e.g. LONG, WIN)..." onkeyup="filterTable()">
                <select id="resultFilter" onchange="filterTable()">
                    <option value="ALL">All Outcomes</option>
                    <option value="WIN">Wins Only</option>
                    <option value="LOSS">Losses Only</option>
                </select>
            </div>
        </div>
        <table id="tradesTable">
            <thead>
                <tr>
                    <th onclick="sortTable(0)">ID</th>
                    <th onclick="sortTable(1)">Signal Time (UTC)</th>
                    <th onclick="sortTable(2)">Side</th>
                    <th onclick="sortTable(3)">Entry</th>
                    <th onclick="sortTable(4)">Stop Loss</th>
                    <th onclick="sortTable(5)">Take Profit</th>
                    <th onclick="sortTable(6)">Exit</th>
                    <th onclick="sortTable(7)">R-Mult</th>
                    <th onclick="sortTable(8)">Gross P&L</th>
                    <th onclick="sortTable(9)">Net P&L</th>
                    <th onclick="sortTable(10)">Outcome</th>
                    <th onclick="sortTable(11)">Exit Reason</th>
                </tr>
            </thead>
            <tbody>
                <!-- Populated via Javascript -->
            </tbody>
        </table>
    </div>

    <script>
        const trades = {trades_json};
        const stats = {stats_json};

        // Chart styling defaults
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.borderColor = '#26334d';

        // 1. Equity Curve
        let currentBalance = stats.initial_balance;
        const equityPoints = [currentBalance];
        const labels = ['0'];

        trades.forEach((t, index) => {{
            currentBalance += t.net_pnl;
            equityPoints.push(currentBalance);
            labels.push(`T${{index + 1}}`);
        }});

        new Chart(document.getElementById('equityChart'), {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [{{
                    label: 'Account Equity ($)',
                    data: equityPoints,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.15,
                    borderWidth: 2,
                    pointRadius: trades.length > 100 ? 0 : 3
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    tooltip: {{
                        callbacks: {{
                            label: (ctx) => `Equity: $${{ctx.parsed.y.toLocaleString('en-US', {{minimumFractionDigits: 2}})}}`
                        }}
                    }}
                }},
                scales: {{
                    y: {{ grid: {{ color: '#1e293b' }} }},
                    x: {{ grid: {{ display: false }} }}
                }}
            }}
        }});

        // 2. P&L Bar Chart (R multiples)
        new Chart(document.getElementById('pnlBarChart'), {{
            type: 'bar',
            data: {{
                labels: trades.map(t => `#${{t.trade_id}}`),
                datasets: [{{
                    label: 'R Multiple',
                    data: trades.map(t => t.R_multiple),
                    backgroundColor: trades.map(t => t.R_multiple >= 0 ? 'rgba(16, 185, 129, 0.7)' : 'rgba(239, 68, 68, 0.7)'),
                    borderColor: trades.map(t => t.R_multiple >= 0 ? '#10b981' : '#ef4444'),
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{ title: {{ display: true, text: 'R Multiple' }}, grid: {{ color: '#1e293b' }} }},
                    x: {{ display: trades.length <= 50 }}
                }}
            }}
        }});

        // 3. Win / Loss Chart
        new Chart(document.getElementById('winLossChart'), {{
            type: 'doughnut',
            data: {{
                labels: ['Wins', 'Losses', 'Breakeven'],
                datasets: [{{
                    data: [stats.winning_trades, stats.losing_trades, stats.breakeven_trades],
                    backgroundColor: ['#10b981', '#ef4444', '#64748b'],
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'bottom' }}
                }}
            }}
        }});

        // 4. Long vs Short Performance
        const longTrades = trades.filter(t => t.side === 'LONG');
        const shortTrades = trades.filter(t => t.side === 'SHORT');
        const longNet = longTrades.reduce((acc, t) => acc + t.net_pnl, 0);
        const shortNet = shortTrades.reduce((acc, t) => acc + t.net_pnl, 0);

        new Chart(document.getElementById('sideChart'), {{
            type: 'bar',
            data: {{
                labels: ['Long Trades', 'Short Trades'],
                datasets: [{{
                    label: 'Net P&L ($)',
                    data: [longNet, shortNet],
                    backgroundColor: ['rgba(59, 130, 246, 0.8)', 'rgba(245, 158, 11, 0.8)']
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{ grid: {{ color: '#1e293b' }} }}
                }}
            }}
        }});

        // 5. Exit Reason Breakdown
        const exitReasons = {{}};
        trades.forEach(t => {{
            exitReasons[t.exit_reason] = (exitReasons[t.exit_reason] || 0) + 1;
        }});

        new Chart(document.getElementById('exitReasonChart'), {{
            type: 'pie',
            data: {{
                labels: Object.keys(exitReasons),
                datasets: [{{
                    data: Object.values(exitReasons),
                    backgroundColor: ['#10b981', '#ef4444', '#f59e0b', '#8b5cf6', '#64748b']
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ position: 'bottom' }} }}
            }}
        }});

        // 6. Hour-of-Day P&L
        const hourPnl = Array(24).fill(0);
        trades.forEach(t => {{
            const hour = new Date(t.entry_time).getUTCHours();
            if (!isNaN(hour)) {{
                hourPnl[hour] += t.net_pnl;
            }}
        }});

        new Chart(document.getElementById('hourlyChart'), {{
            type: 'bar',
            data: {{
                labels: Array.from({{length: 24}}, (_, i) => `${{String(i).padStart(2, '0')}}:00 UTC`),
                datasets: [{{
                    label: 'Net P&L ($)',
                    data: hourPnl,
                    backgroundColor: hourPnl.map(v => v >= 0 ? 'rgba(16, 185, 129, 0.7)' : 'rgba(239, 68, 68, 0.7)'),
                    borderColor: hourPnl.map(v => v >= 0 ? '#10b981' : '#ef4444'),
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{ grid: {{ color: '#1e293b' }} }},
                    x: {{ grid: {{ color: '#1e293b' }} }}
                }}
            }}
        }});

        // 7. Populate Trade Ledger Table
        const tbody = document.querySelector('#tradesTable tbody');
        trades.forEach(t => {{
            const tr = document.createElement('tr');
            const sideClass = t.side === 'LONG' ? 'tag-long' : 'tag-short';
            const resClass = t.result === 'WIN' ? 'tag-win' : (t.result === 'LOSS' ? 'tag-loss' : '');
            const pnlClass = t.net_pnl >= 0 ? 'text-green' : 'text-red';

            tr.innerHTML = `
                <td>#${{t.trade_id}}</td>
                <td>${{t.signal_time.replace('T', ' ').substring(0, 19)}}</td>
                <td><span class="tag ${{sideClass}}">${{t.side}}</span></td>
                <td>${{t.entry_price.toFixed(2)}}</td>
                <td>${{t.stop_price.toFixed(2)}}</td>
                <td>${{t.target_price.toFixed(2)}}</td>
                <td>${{t.exit_price !== null ? t.exit_price.toFixed(2) : '-'}}</td>
                <td class="${{pnlClass}}">${{t.R_multiple > 0 ? '+' : ''}}${{t.R_multiple.toFixed(2)}}R</td>
                <td>$${{t.gross_pnl.toFixed(2)}}</td>
                <td class="${{pnlClass}}" style="font-weight:600">$${{t.net_pnl.toFixed(2)}}</td>
                <td><span class="tag ${{resClass}}">${{t.result}}</span></td>
                <td><code>${{t.exit_reason}}</code></td>
            `;
            tbody.appendChild(tr);
        }});

        // Search & Filter Table
        function filterTable() {{
            const search = document.getElementById('tableSearch').value.toUpperCase();
            const outcome = document.getElementById('resultFilter').value;
            const rows = document.querySelectorAll('#tradesTable tbody tr');

            rows.forEach(r => {{
                const text = r.textContent.toUpperCase();
                const matchesSearch = text.includes(search);
                const matchesOutcome = outcome === 'ALL' || text.includes(outcome);
                r.style.display = (matchesSearch && matchesOutcome) ? '' : 'none';
            }});
        }}

        // Sort Table
        let sortDirection = {{}};
        function sortTable(colIndex) {{
            const table = document.getElementById("tradesTable");
            let rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
            switching = true;
            dir = sortDirection[colIndex] === "asc" ? "desc" : "asc";
            sortDirection[colIndex] = dir;
            
            while (switching) {{
                switching = false;
                rows = table.rows;
                for (i = 1; i < (rows.length - 1); i++) {{
                    shouldSwitch = false;
                    x = rows[i].getElementsByTagName("TD")[colIndex];
                    y = rows[i + 1].getElementsByTagName("TD")[colIndex];
                    let xVal = x.innerText.replace(/[$,R#+]/g, '');
                    let yVal = y.innerText.replace(/[$,R#+]/g, '');
                    let xNum = parseFloat(xVal);
                    let yNum = parseFloat(yVal);
                    
                    if (!isNaN(xNum) && !isNaN(yNum)) {{
                        if (dir === "asc" ? xNum > yNum : xNum < yNum) {{
                            shouldSwitch = true;
                            break;
                        }}
                    }} else {{
                        if (dir === "asc" ? xVal.toLowerCase() > yVal.toLowerCase() : xVal.toLowerCase() < yVal.toLowerCase()) {{
                            shouldSwitch = true;
                            break;
                        }}
                    }}
                }}
                if (shouldSwitch) {{
                    rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                    switching = true;
                    switchcount++;
                }}
            }}
        }}
    </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"[REPORT] Interactive HTML Dashboard successfully generated at: {output_path}")


# ----------------------------------------------------------------------
# 5. Output Serialization & CLI Main
# ----------------------------------------------------------------------
def export_results(
    trades: List[Dict[str, Any]],
    stats: Dict[str, Any],
    audit_log: List[Dict[str, Any]],
    output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. trades.csv
    trades_path = os.path.join(output_dir, "trades.csv")
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(trades_path, index=False)
    else:
        # Create empty template with required headers
        headers = [
            "trade_id", "date", "side", "signal_time", "entry_time", "exit_time",
            "entry_price", "stop_price", "target_price", "exit_price",
            "risk_points", "reward_points", "R_multiple", "gross_pnl", "cost",
            "net_pnl", "result", "exit_reason", "signal_bar_index", "entry_bar_index", "exit_bar_index"
        ]
        pd.DataFrame(columns=headers).to_csv(trades_path, index=False)
    print(f"[EXPORT] Trades ledger saved to: {trades_path}")

    # 2. statistics.json
    stats_path = os.path.join(output_dir, "statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
    print(f"[EXPORT] Statistics saved to: {stats_path}")

    # 3. audit.csv
    audit_path = os.path.join(output_dir, "audit.csv")
    if audit_log:
        audit_df = pd.DataFrame(audit_log)
        audit_df.to_csv(audit_path, index=False)
    else:
        headers = ["signal_bar_index", "signal_time", "available_data_until", "signal_reason", "expected_entry_bar", "side", "stop_loss"]
        pd.DataFrame(columns=headers).to_csv(audit_path, index=False)
    print(f"[EXPORT] Audit trail saved to: {audit_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="US100 M1 Price-Action Scalping Backtester")
    parser.add_argument("--csv", type=str, required=True, help="Path to M1 OHLC CSV file")
    parser.add_argument("--initial-balance", type=float, default=10000.0, help="Initial capital in USD")
    parser.add_argument("--risk-per-trade", type=float, default=100.0, help="Fixed risk per trade in USD (1R)")
    parser.add_argument("--risk-reward", type=float, default=2.0, help="Risk-to-reward ratio multiplier")
    parser.add_argument("--spread", type=float, default=1.0, help="Configured spread in index points")
    parser.add_argument("--slippage", type=float, default=0.5, help="Configured slippage in index points")
    parser.add_argument("--commission", type=float, default=0.0, help="Commission per trade in USD")
    parser.add_argument("--or-candles", type=int, default=5, help="Opening Range candles count")
    parser.add_argument("--max-retest", type=int, default=3, help="Max retest candles after breakout")
    parser.add_argument("--audit", action="store_true", default=True, help="Enable signal audit logging")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save CSV and JSON results")
    parser.add_argument("--html-output", type=str, default="backtest.html", help="Path for HTML dashboard output")
    return parser.parse_args()


def main():
    args = parse_arguments()

    config = BacktestConfig(
        csv_path=args.csv,
        initial_balance=args.initial_balance,
        risk_per_trade=args.risk_per_trade,
        risk_reward_ratio=args.risk_reward,
        spread_points=args.spread,
        slippage_points=args.slippage,
        commission_per_trade=args.commission,
        or_candles_count=args.or_candles,
        max_retest_candles=args.max_retest,
        audit_mode=args.audit,
        output_dir=args.output_dir,
        html_output=args.html_output,
    )

    print("=" * 70)
    print(" US100 M1 PRICE-ACTION SCALPING BACKTESTER (ZERO LOOK-AHEAD)")
    print("=" * 70)

    # 1. Load data
    df = load_and_validate_data(config.csv_path)

    # 2. Run Engine
    engine = BacktestEngine(df, config)
    trades, stats, audit_log = engine.run()

    # 3. Export Trades, Stats, Audit
    export_results(trades, stats, audit_log, config.output_dir)

    # 4. Generate Interactive HTML Dashboard
    generate_html_dashboard(trades, stats, df, config, config.html_output)

    # 5. Print Execution Summary
    print("\n" + "=" * 70)
    print(" BACKTEST EXECUTION SUMMARY")
    print("=" * 70)
    print(f" Total M1 Candles Processed: {stats['candle_count']:,}")
    print(f" Total Trades Executed:      {stats['total_trades']}")
    print(f" Win Rate:                   {stats['win_rate']:.2f}% ({stats['winning_trades']}W / {stats['losing_trades']}L / {stats['breakeven_trades']}BE)")
    print(f" Profit Factor:              {stats['profit_factor']:.2f}")
    print(f" Average R-Multiple:         {stats['average_r']:+.2f}R")
    print(f" Gross P&L:                  ${stats['gross_pnl']:,.2f}")
    print(f" Total Trading Costs:        ${stats['total_costs']:,.2f}")
    print(f" Net P&L:                    ${stats['net_pnl']:,.2f}")
    print(f" Return on Capital:          {stats['return_pct']:.2f}%")
    print(f" Max Drawdown:               {stats['max_drawdown_pct']:.2f}% (${stats['max_drawdown_dollars']:,.2f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
