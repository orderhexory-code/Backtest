#!/usr/bin/env python3
"""
US100 M1 Price-Action Scalping Backtester
Strategy: M1 Opening Range Breakout + Retest with Cost-to-Cost (Breakeven SL)
Zero Look-Ahead Bias | Production Ready | Minimal TSX-Style UI
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
# Configuration
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
        enable_c2c: bool = True,
        c2c_trigger_r: float = 1.0,
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
        self.enable_c2c = bool(enable_c2c)
        self.c2c_trigger_r = float(c2c_trigger_r)
        self.audit_mode = bool(audit_mode)
        self.output_dir = output_dir
        self.html_output = html_output


# ----------------------------------------------------------------------
# 1. Data Ingestion & Strict Validation
# ----------------------------------------------------------------------
def load_and_validate_data(csv_path: str) -> pd.DataFrame:
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

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop invalid/malformed OHLC rows
    df = df.dropna(subset=required_columns).copy()
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    df = df[
        (df["high"] >= df["low"])
        & (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
    ]

    try:
        df["timestamp_num"] = pd.to_numeric(df["timestamp"], errors="coerce")
        if df["timestamp_num"].notnull().all():
            if df["timestamp_num"].iloc[0] > 1e11:
                df["datetime"] = pd.to_datetime(df["timestamp_num"], unit="ms", utc=True)
            else:
                df["datetime"] = pd.to_datetime(df["timestamp_num"], unit="s", utc=True)
        else:
            df["datetime"] = pd.to_datetime(df["timestamp"], utc=True)
    except Exception as e:
        raise ValueError(f"Failed to parse datetime timestamps: {e}")

    df = df.dropna(subset=["datetime"]).copy()
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    final_len = len(df)
    print(f"[DATA] Loaded {final_len:,} valid M1 bars. Dropped {initial_len - final_len:,} invalid/duplicate rows.")
    print(f"[DATA] Window: {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")
    return df


# ----------------------------------------------------------------------
# 2. Sequential Zero Look-Ahead Event-Driven Execution Engine
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
            print("[WARN] Insufficient candle history.")
            stats = calculate_statistics(self.trades, self.cfg, self.df)
            return self.trades, stats, self.audit_log

        opens = self.df["open"].to_numpy()
        highs = self.df["high"].to_numpy()
        lows = self.df["low"].to_numpy()
        closes = self.df["close"].to_numpy()
        datetimes = self.df["datetime"].to_numpy()
        dates = pd.to_datetime(datetimes).date

        current_session_date = None
        session_bar_idx = 0
        or_high = None
        or_low = None
        or_established = False

        setup_state = "IDLE"
        breakout_side = None
        breakout_bar_idx = None
        retest_candles_elapsed = 0
        pending_signal = None

        open_trade: Optional[Dict[str, Any]] = None
        trade_counter = 0

        for i in range(n):
            c_date = dates[i]
            c_open = opens[i]
            c_high = highs[i]
            c_low = lows[i]
            c_close = closes[i]
            c_dt = str(datetimes[i])

            # Session boundary check
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
            # STEP A: Execute Pending Signal at Next Bar Open
            # ----------------------------------------------------------
            if pending_signal is not None and open_trade is None:
                side = pending_signal["side"]
                sig_sl = pending_signal["sl_price"]
                sig_bar = pending_signal["signal_bar_index"]

                if side == "LONG":
                    entry_price = c_open + self.cfg.slippage_points + self.cfg.spread_points
                    risk_points = entry_price - sig_sl
                    if risk_points <= 0:
                        risk_points = 1.0
                    reward_points = risk_points * self.cfg.risk_reward_ratio
                    target_price = entry_price + reward_points
                    c2c_price = entry_price + (risk_points * self.cfg.c2c_trigger_r)
                else:  # SHORT
                    entry_price = c_open - self.cfg.slippage_points
                    risk_points = sig_sl - entry_price
                    if risk_points <= 0:
                        risk_points = 1.0
                    reward_points = risk_points * self.cfg.risk_reward_ratio
                    target_price = entry_price - reward_points
                    c2c_price = entry_price - (risk_points * self.cfg.c2c_trigger_r)

                trade_counter += 1
                open_trade = {
                    "trade_id": trade_counter,
                    "date": str(c_date),
                    "side": side,
                    "signal_time": pending_signal["signal_time"],
                    "entry_time": c_dt,
                    "exit_time": None,
                    "entry_price": round(entry_price, 4),
                    "initial_stop_price": round(sig_sl, 4),
                    "stop_price": round(sig_sl, 4),
                    "target_price": round(target_price, 4),
                    "c2c_trigger_price": round(c2c_price, 4),
                    "is_sl_at_cost": False,
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
            # STEP B: Manage Active Open Trade (Cost-to-Cost Trailing)
            # ----------------------------------------------------------
            if open_trade is not None:
                side = open_trade["side"]
                ep = open_trade["entry_price"]
                sl = open_trade["stop_price"]
                tp = open_trade["target_price"]
                risk_pts = open_trade["risk_points"]
                c2c_trigger = open_trade["c2c_trigger_price"]
                is_sl_at_cost = open_trade["is_sl_at_cost"]

                trade_closed = False
                exit_price = 0.0
                exit_reason = ""
                result = ""

                # Cost-to-Cost (Breakeven) trigger
                if self.cfg.enable_c2c and not is_sl_at_cost:
                    if (side == "LONG" and c_high >= c2c_trigger) or (side == "SHORT" and c_low <= c2c_trigger):
                        open_trade["is_sl_at_cost"] = True
                        open_trade["stop_price"] = ep
                        sl = ep
                        is_sl_at_cost = True

                # Conservative SL/TP evaluation
                if side == "LONG":
                    hit_sl = c_low <= sl
                    hit_tp = c_high >= tp

                    if hit_sl and hit_tp:
                        trade_closed = True
                        exit_price = sl - self.cfg.slippage_points
                        exit_reason = "AMBIGUOUS_SL_TP_SL_FIRST"
                        result = "LOSS" if not is_sl_at_cost else "BREAKEVEN"
                    elif hit_sl:
                        trade_closed = True
                        exit_price = sl - (self.cfg.slippage_points if not is_sl_at_cost else 0.0)
                        exit_reason = "COST_TO_COST_BREAKEVEN" if is_sl_at_cost else "STOP_LOSS"
                        result = "BREAKEVEN" if is_sl_at_cost else "LOSS"
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
                        result = "LOSS" if not is_sl_at_cost else "BREAKEVEN"
                    elif hit_sl:
                        trade_closed = True
                        exit_price = sl + (self.cfg.slippage_points + self.cfg.spread_points if not is_sl_at_cost else 0.0)
                        exit_reason = "COST_TO_COST_BREAKEVEN" if is_sl_at_cost else "STOP_LOSS"
                        result = "BREAKEVEN" if is_sl_at_cost else "LOSS"
                    elif hit_tp:
                        trade_closed = True
                        exit_price = tp
                        exit_reason = "TAKE_PROFIT"
                        result = "WIN"

                # End of dataset exit
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

                    price_diff = (exit_price - ep) if side == "LONG" else (ep - exit_price)
                    r_multiple = price_diff / risk_pts if risk_pts > 0 else 0.0
                    open_trade["R_multiple"] = round(r_multiple, 4)

                    gross_pnl = r_multiple * self.cfg.risk_per_trade
                    net_pnl = gross_pnl - open_trade["cost"]
                    open_trade["gross_pnl"] = round(gross_pnl, 2)
                    open_trade["net_pnl"] = round(net_pnl, 2)

                    self.trades.append(open_trade)
                    open_trade = None
                    setup_state = "SEARCHING_BREAKOUT"

                continue

            # ----------------------------------------------------------
            # STEP C: Opening Range Construction (Strictly Historical)
            # ----------------------------------------------------------
            if session_bar_idx < self.cfg.or_candles_count:
                continue
            elif session_bar_idx == self.cfg.or_candles_count:
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
            # STEP D: Setup Detection & Signal Scan
            # ----------------------------------------------------------
            if setup_state == "SEARCHING_BREAKOUT":
                if c_close > or_high:
                    setup_state = "WAITING_RETEST"
                    breakout_side = "LONG"
                    breakout_bar_idx = i
                    retest_candles_elapsed = 0
                elif c_close < or_low:
                    setup_state = "WAITING_RETEST"
                    breakout_side = "SHORT"
                    breakout_bar_idx = i
                    retest_candles_elapsed = 0

            elif setup_state == "WAITING_RETEST":
                retest_candles_elapsed += 1
                if retest_candles_elapsed > self.cfg.max_retest_candles:
                    setup_state = "SEARCHING_BREAKOUT"
                    breakout_side = None
                    breakout_bar_idx = None
                    continue

                if breakout_side == "LONG":
                    if c_low < or_low:
                        setup_state = "SEARCHING_BREAKOUT"
                        continue

                    is_retest = c_low <= or_high * 1.0005
                    is_bullish_rejection = (c_close > c_open) and (c_close >= or_high)

                    if is_retest and is_bullish_rejection:
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
                    if c_high > or_high:
                        setup_state = "SEARCHING_BREAKOUT"
                        continue

                    is_retest = c_high >= or_low * 0.9995
                    is_bearish_rejection = (c_close < c_open) and (c_close <= or_low)

                    if is_retest and is_bearish_rejection:
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
# 3. Analytics Calculation
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
            "date_range": [str(df["datetime"].iloc[0]), str(df["datetime"].iloc[-1])] if len(df) > 0 else ["", ""],
        }

    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] < 0]
    bes = [t for t in trades if t["net_pnl"] == 0 or t["result"] == "BREAKEVEN"]

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
    expectancy = (win_rate / 100.0 * avg_win) - ((1.0 - (win_rate / 100.0)) * avg_loss)

    equity = initial_balance
    peak = initial_balance
    max_dd_dollars = 0.0
    max_dd_pct = 0.0

    for t in trades:
        equity += t["net_pnl"]
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

    longs = [t for t in trades if t["side"] == "LONG"]
    shorts = [t for t in trades if t["side"] == "SHORT"]
    long_wins = [t for t in longs if t["net_pnl"] > 0]
    short_wins = [t for t in shorts if t["net_pnl"] > 0]

    long_win_rate = (len(long_wins) / len(longs) * 100.0) if longs else 0.0
    short_win_rate = (len(short_wins) / len(shorts) * 100.0) if shorts else 0.0

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
# 4. Minimalist TSX / Shadcn-Style Dashboard HTML Generator
# ----------------------------------------------------------------------
def generate_html_dashboard(
    trades: List[Dict[str, Any]],
    stats: Dict[str, Any],
    df: pd.DataFrame,
    cfg: BacktestConfig,
    output_path: str,
):
    trades_json = json.dumps(trades)
    stats_json = json.dumps(stats)

    html_template = f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>US100 M1 Scalping — Analytics</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #09090b;
            --card: #121215;
            --card-hover: #18181b;
            --border: #27272a;
            --border-light: #3f3f46;
            --text-main: #f4f4f5;
            --text-muted: #a1a1aa;
            --text-subtle: #71717a;
            --green: #10b981;
            --green-subtle: rgba(16, 185, 129, 0.12);
            --red: #f43f5e;
            --red-subtle: rgba(244, 63, 94, 0.12);
            --amber: #f59e0b;
            --amber-subtle: rgba(245, 158, 11, 0.12);
            --blue: #3b82f6;
            --blue-subtle: rgba(59, 130, 246, 0.12);
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg);
            color: var(--text-main);
            font-family: 'Inter', -apple-system, sans-serif;
            padding: 32px 24px;
            min-height: 100vh;
            -webkit-font-smoothing: antialiased;
        }}
        .container {{
            max-width: 1380px;
            margin: 0 auto;
        }}
        .font-mono {{ font-family: 'Geist Mono', monospace; }}
        
        /* Navbar / Header */
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 28px;
            flex-wrap: wrap;
            gap: 16px;
        }}
        .header-title-wrap {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .header-title {{
            font-size: 20px;
            font-weight: 600;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .badge {{
            font-size: 11px;
            font-weight: 500;
            padding: 2px 8px;
            border-radius: 9999px;
            border: 1px solid var(--border);
            background-color: #18181b;
            color: var(--text-muted);
            letter-spacing: 0.02em;
        }}
        .badge-c2c {{
            border-color: rgba(245, 158, 11, 0.3);
            background-color: var(--amber-subtle);
            color: var(--amber);
        }}
        .header-subtitle {{
            font-size: 13px;
            color: var(--text-subtle);
        }}
        .header-meta {{
            font-size: 12px;
            color: var(--text-muted);
            display: flex;
            gap: 12px;
            align-items: center;
            background: var(--card);
            border: 1px solid var(--border);
            padding: 6px 14px;
            border-radius: 8px;
        }}

        /* KPI Cards Grid (Tailwind/Shadcn style) */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 12px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        .kpi-card:hover {{
            border-color: var(--border-light);
            background: var(--card-hover);
        }}
        .kpi-label {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-subtle);
            margin-bottom: 8px;
            font-weight: 500;
        }}
        .kpi-value {{
            font-size: 22px;
            font-weight: 600;
            letter-spacing: -0.02em;
        }}
        .kpi-subtext {{
            font-size: 11px;
            color: var(--text-subtle);
            margin-top: 4px;
        }}

        /* Charts Layout */
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        @media (max-width: 1024px) {{
            .charts-grid {{ grid-template-columns: 1fr; }}
        }}
        .chart-box {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
        }}
        .chart-box.full-width {{
            grid-column: 1 / -1;
        }}
        .chart-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 18px;
        }}
        .chart-title {{
            font-size: 13px;
            font-weight: 600;
            letter-spacing: -0.01em;
            color: var(--text-main);
        }}

        /* Clean Monospace Table */
        .table-section {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            overflow-x: auto;
        }}
        .table-toolbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .input-search, .select-filter {{
            background: #09090b;
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 7px 12px;
            border-radius: 6px;
            font-size: 12px;
            outline: none;
            transition: border-color 0.15s ease;
        }}
        .input-search:focus, .select-filter:focus {{
            border-color: var(--border-light);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            text-align: left;
        }}
        th {{
            color: var(--text-subtle);
            font-weight: 500;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
            cursor: pointer;
            user-select: none;
        }}
        th:hover {{ color: var(--text-muted); }}
        td {{
            padding: 11px 12px;
            border-bottom: 1px solid rgba(39, 39, 42, 0.6);
            color: var(--text-muted);
        }}
        tr:hover td {{
            background-color: rgba(255, 255, 255, 0.015);
            color: var(--text-main);
        }}
        .tag {{
            display: inline-flex;
            align-items: center;
            padding: 2px 7px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 0.02em;
        }}
        .tag-win {{ background: var(--green-subtle); color: var(--green); }}
        .tag-loss {{ background: var(--red-subtle); color: var(--red); }}
        .tag-be {{ background: var(--amber-subtle); color: var(--amber); }}
        .tag-long {{ background: var(--blue-subtle); color: var(--blue); }}
        .tag-short {{ background: var(--amber-subtle); color: var(--amber); }}
        .text-green {{ color: var(--green); }}
        .text-red {{ color: var(--red); }}
        .text-amber {{ color: var(--amber); }}
    </style>
</head>
<body>

<div class="container">
    <!-- Header -->
    <header class="header">
        <div class="header-title-wrap">
            <div class="header-title">
                <span>US100 M1 Scalper</span>
                <span class="badge">v1.1</span>
                <span class="badge badge-c2c">C2C Breakeven Active</span>
            </div>
            <div class="header-subtitle">
                Deterministic M1 ORB + Retest | Strict 0 Look-Ahead Bias | Automated GitHub Pages Execution
            </div>
        </div>
        <div class="header-meta font-mono">
            <span><strong>Dataset:</strong> {os.path.basename(cfg.csv_path)}</span>
            <span>•</span>
            <span>{stats['candle_count']:,} Bars</span>
            <span>•</span>
            <span>Spread {cfg.spread_points}pt</span>
            <span>•</span>
            <span>Slip {cfg.slippage_points}pt</span>
        </div>
    </header>

    <!-- KPI Grid -->
    <section class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Net Profit</div>
            <div class="kpi-value font-mono {'text-green' if stats['net_pnl'] >= 0 else 'text-red'}">
                ${stats['net_pnl']:,.2f}
            </div>
            <div class="kpi-subtext font-mono">Gross: ${stats['gross_pnl']:,.2f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Return</div>
            <div class="kpi-value font-mono {'text-green' if stats['return_pct'] >= 0 else 'text-red'}">
                {stats['return_pct']:.2f}%
            </div>
            <div class="kpi-subtext font-mono">Start: ${stats['initial_balance']:,.0f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Win Rate</div>
            <div class="kpi-value font-mono text-green">{stats['win_rate']:.1f}%</div>
            <div class="kpi-subtext font-mono">{stats['winning_trades']}W / {stats['losing_trades']}L</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">C2C Saves</div>
            <div class="kpi-value font-mono text-amber">{stats['breakeven_trades']}</div>
            <div class="kpi-subtext font-mono">Breakeven Exits</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Profit Factor</div>
            <div class="kpi-value font-mono {'text-green' if stats['profit_factor'] >= 1.0 else 'text-red'}">
                {stats['profit_factor']:.2f}
            </div>
            <div class="kpi-subtext font-mono">Exp: ${stats['expectancy']:.2f}/tr</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Max Drawdown</div>
            <div class="kpi-value font-mono text-red">
                {stats['max_drawdown_pct']:.2f}%
            </div>
            <div class="kpi-subtext font-mono">${stats['max_drawdown_dollars']:,.2f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Total Trades</div>
            <div class="kpi-value font-mono">{stats['total_trades']}</div>
            <div class="kpi-subtext font-mono">Avg R: {stats['average_r']:+.2f}R</div>
        </div>
    </section>

    <!-- Charts -->
    <section class="charts-grid">
        <div class="chart-box full-width">
            <div class="chart-header">
                <div class="chart-title">Equity Growth & Capital Preservation</div>
                <div class="badge font-mono">USD ($)</div>
            </div>
            <div style="height: 280px;"><canvas id="equityChart"></canvas></div>
        </div>

        <div class="chart-box">
            <div class="chart-header">
                <div class="chart-title">Trade Outcome Ledger (R-Multiple)</div>
            </div>
            <div style="height: 240px;"><canvas id="pnlBarChart"></canvas></div>
        </div>

        <div class="chart-box">
            <div class="chart-header">
                <div class="chart-title">Outcome Distribution</div>
            </div>
            <div style="height: 240px;"><canvas id="winLossChart"></canvas></div>
        </div>

        <div class="chart-box">
            <div class="chart-header">
                <div class="chart-title">Exit Reason Breakdown (Audited)</div>
            </div>
            <div style="height: 240px;"><canvas id="exitReasonChart"></canvas></div>
        </div>

        <div class="chart-box">
            <div class="chart-header">
                <div class="chart-title">Long vs Short Performance</div>
            </div>
            <div style="height: 240px;"><canvas id="sideChart"></canvas></div>
        </div>
    </section>

    <!-- Table -->
    <section class="table-section">
        <div class="table-toolbar">
            <div class="chart-title">Trade Audit Log</div>
            <div style="display:flex; gap: 8px;">
                <input type="text" id="tableSearch" class="input-search font-mono" placeholder="Search ID, Long, Win..." onkeyup="filterTable()">
                <select id="resultFilter" class="select-filter font-mono" onchange="filterTable()">
                    <option value="ALL">All Outcomes</option>
                    <option value="WIN">Wins</option>
                    <option value="LOSS">Losses</option>
                    <option value="BREAKEVEN">C2C Breakevens</option>
                </select>
            </div>
        </div>
        <table id="tradesTable">
            <thead>
                <tr>
                    <th onclick="sortTable(0)">ID</th>
                    <th onclick="sortTable(1)">Signal Time</th>
                    <th onclick="sortTable(2)">Side</th>
                    <th onclick="sortTable(3)">Entry</th>
                    <th onclick="sortTable(4)">Stop</th>
                    <th onclick="sortTable(5)">C2C Level</th>
                    <th onclick="sortTable(6)">Target</th>
                    <th onclick="sortTable(7)">Exit</th>
                    <th onclick="sortTable(8)">R</th>
                    <th onclick="sortTable(9)">Net P&L</th>
                    <th onclick="sortTable(10)">Outcome</th>
                    <th onclick="sortTable(11)">Exit Reason</th>
                </tr>
            </thead>
            <tbody class="font-mono"></tbody>
        </table>
    </section>
</div>

<script>
    const trades = {trades_json};
    const stats = {stats_json};

    Chart.defaults.color = '#71717a';
    Chart.defaults.borderColor = '#27272a';
    Chart.defaults.font.family = "'Geist Mono', monospace";
    Chart.defaults.font.size = 11;

    // Equity Curve
    let currentBalance = stats.initial_balance;
    const equityPoints = [currentBalance];
    const labels = ['0'];
    trades.forEach((t, i) => {{
        currentBalance += t.net_pnl;
        equityPoints.push(currentBalance);
        labels.push(`T${{i + 1}}`);
    }});

    new Chart(document.getElementById('equityChart'), {{
        type: 'line',
        data: {{
            labels: labels,
            datasets: [{{
                label: 'Equity ($)',
                data: equityPoints,
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.05)',
                borderWidth: 1.75,
                fill: true,
                tension: 0.1,
                pointRadius: trades.length > 80 ? 0 : 2
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{ legend: {{ display: false }} }},
            scales: {{
                x: {{ grid: {{ display: false }} }},
                y: {{ grid: {{ color: '#18181b' }} }}
            }}
        }}
    }});

    // PnL Bar Chart
    new Chart(document.getElementById('pnlBarChart'), {{
        type: 'bar',
        data: {{
            labels: trades.map(t => `#${{t.trade_id}}`),
            datasets: [{{
                data: trades.map(t => t.R_multiple),
                backgroundColor: trades.map(t => t.R_multiple > 0 ? '#10b981' : (t.R_multiple < 0 ? '#f43f5e' : '#f59e0b')),
                borderRadius: 2
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{ legend: {{ display: false }} }},
            scales: {{
                x: {{ display: trades.length <= 40, grid: {{ display: false }} }},
                y: {{ grid: {{ color: '#18181b' }} }}
            }}
        }}
    }});

    // Win Loss Doughnut
    new Chart(document.getElementById('winLossChart'), {{
        type: 'doughnut',
        data: {{
            labels: ['Win', 'Loss', 'C2C Breakeven'],
            datasets: [{{
                data: [stats.winning_trades, stats.losing_trades, stats.breakeven_trades],
                backgroundColor: ['#10b981', '#f43f5e', '#f59e0b'],
                borderWidth: 0
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{ legend: {{ position: 'bottom' }} }}
        }}
    }});

    // Exit Reason
    const exitReasons = {{}};
    trades.forEach(t => exitReasons[t.exit_reason] = (exitReasons[t.exit_reason] || 0) + 1);
    new Chart(document.getElementById('exitReasonChart'), {{
        type: 'doughnut',
        data: {{
            labels: Object.keys(exitReasons),
            datasets: [{{
                data: Object.values(exitReasons),
                backgroundColor: ['#10b981', '#f43f5e', '#f59e0b', '#3b82f6', '#71717a'],
                borderWidth: 0
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{ legend: {{ position: 'bottom' }} }}
        }}
    }});

    // Long vs Short
    const longNet = trades.filter(t => t.side === 'LONG').reduce((acc, t) => acc + t.net_pnl, 0);
    const shortNet = trades.filter(t => t.side === 'SHORT').reduce((acc, t) => acc + t.net_pnl, 0);
    new Chart(document.getElementById('sideChart'), {{
        type: 'bar',
        data: {{
            labels: ['Long', 'Short'],
            datasets: [{{
                data: [longNet, shortNet],
                backgroundColor: ['#3b82f6', '#f59e0b'],
                borderRadius: 4
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{ legend: {{ display: false }} }},
            scales: {{
                x: {{ grid: {{ display: false }} }},
                y: {{ grid: {{ color: '#18181b' }} }}
            }}
        }}
    }});

    // Table Rendering
    const tbody = document.querySelector('#tradesTable tbody');
    trades.forEach(t => {{
        const tr = document.createElement('tr');
        const sideTag = t.side === 'LONG' ? 'tag-long' : 'tag-short';
        const resTag = t.result === 'WIN' ? 'tag-win' : (t.result === 'LOSS' ? 'tag-loss' : 'tag-be');
        const pnlColor = t.net_pnl > 0 ? 'text-green' : (t.net_pnl < 0 ? 'text-red' : 'text-amber');

        tr.innerHTML = `
            <td>#${{t.trade_id}}</td>
            <td>${{t.signal_time.replace('T', ' ').substring(0, 19)}}</td>
            <td><span class="tag ${{sideTag}}">${{t.side}}</span></td>
            <td>${{t.entry_price.toFixed(2)}}</td>
            <td>${{t.stop_price.toFixed(2)}}</td>
            <td>${{t.c2c_trigger_price.toFixed(2)}}</td>
            <td>${{t.target_price.toFixed(2)}}</td>
            <td>${{t.exit_price !== null ? t.exit_price.toFixed(2) : '-'}}</td>
            <td class="${{pnlColor}}">${{t.R_multiple > 0 ? '+' : ''}}${{t.R_multiple.toFixed(2)}}R</td>
            <td class="${{pnlColor}}" style="font-weight:600">$${{t.net_pnl.toFixed(2)}}</td>
            <td><span class="tag ${{resTag}}">${{t.result}}</span></td>
            <td><code style="font-size:10px; color:#a1a1aa">${{t.exit_reason}}</code></td>
        `;
        tbody.appendChild(tr);
    }});

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

    let sortDirection = {{}};
    function sortTable(colIndex) {{
        const table = document.getElementById("tradesTable");
        let rows = table.rows, switching = true, dir = sortDirection[colIndex] === "asc" ? "desc" : "asc";
        sortDirection[colIndex] = dir;
        while (switching) {{
            switching = false;
            for (let i = 1; i < (rows.length - 1); i++) {{
                let shouldSwitch = false;
                let x = rows[i].getElementsByTagName("TD")[colIndex];
                let y = rows[i + 1].getElementsByTagName("TD")[colIndex];
                let xVal = parseFloat(x.innerText.replace(/[$,R#+]/g, '')) || x.innerText.toLowerCase();
                let yVal = parseFloat(y.innerText.replace(/[$,R#+]/g, '')) || y.innerText.toLowerCase();
                if (dir === "asc" ? xVal > yVal : xVal < yVal) {{
                    rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                    switching = true;
                    break;
                }}
            }}
        }}
    }}
</script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"[REPORT] Minimalist Dashboard written to: {output_path}")


# ----------------------------------------------------------------------
# 5. CLI & Execution Entry Point
# ----------------------------------------------------------------------
def export_results(trades: List[Dict[str, Any]], stats: Dict[str, Any], audit_log: List[Dict[str, Any]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    trades_path = os.path.join(output_dir, "trades.csv")
    if trades:
        pd.DataFrame(trades).to_csv(trades_path, index=False)
    else:
        headers = [
            "trade_id", "date", "side", "signal_time", "entry_time", "exit_time",
            "entry_price", "initial_stop_price", "stop_price", "c2c_trigger_price", "target_price", "exit_price",
            "risk_points", "reward_points", "R_multiple", "gross_pnl", "cost",
            "net_pnl", "result", "exit_reason", "signal_bar_index", "entry_bar_index", "exit_bar_index"
        ]
        pd.DataFrame(columns=headers).to_csv(trades_path, index=False)

    with open(os.path.join(output_dir, "statistics.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    audit_path = os.path.join(output_dir, "audit.csv")
    if audit_log:
        pd.DataFrame(audit_log).to_csv(audit_path, index=False)
    else:
        headers = ["signal_bar_index", "signal_time", "available_data_until", "signal_reason", "expected_entry_bar", "side", "stop_loss"]
        pd.DataFrame(columns=headers).to_csv(audit_path, index=False)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="US100 M1 Scalper Backtester")
    parser.add_argument("--csv", type=str, required=True, help="Path to M1 OHLC CSV file")
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--risk-per-trade", type=float, default=100.0)
    parser.add_argument("--risk-reward", type=float, default=2.0)
    parser.add_argument("--spread", type=float, default=1.0)
    parser.add_argument("--slippage", type=float, default=0.5)
    parser.add_argument("--commission", type=float, default=0.0)
    parser.add_argument("--or-candles", type=int, default=5)
    parser.add_argument("--max-retest", type=int, default=3)
    parser.add_argument("--enable-c2c", action="store_true", default=True)
    parser.add_argument("--c2c-trigger-r", type=float, default=1.0)
    parser.add_argument("--audit", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--html-output", type=str, default="backtest.html")
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
        enable_c2c=args.enable_c2c,
        c2c_trigger_r=args.c2c_trigger_r,
        audit_mode=args.audit,
        output_dir=args.output_dir,
        html_output=args.html_output,
    )

    df = load_and_validate_data(config.csv_path)
    engine = BacktestEngine(df, config)
    trades, stats, audit_log = engine.run()

    export_results(trades, stats, audit_log, config.output_dir)
    generate_html_dashboard(trades, stats, df, config, config.html_output)

    print("\n" + "=" * 60)
    print(f" Net P&L:      ${stats['net_pnl']:,.2f} ({stats['return_pct']:.2f}%)")
    print(f" Win Rate:     {stats['win_rate']:.1f}% ({stats['winning_trades']}W / {stats['losing_trades']}L / {stats['breakeven_trades']} C2C)")
    print(f" Max Drawdown: {stats['max_drawdown_pct']:.2f}% (${stats['max_drawdown_dollars']:,.2f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
