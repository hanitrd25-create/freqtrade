#!/usr/bin/env python3
"""Bulk backtesting helper.

Iterates over discovered strategy classes (subclasses of IStrategy) and runs backtests
across multiple timeframes and locally downloaded spot USDT pairs for which data exists.

Generates per (strategy,timeframe) trade export and a consolidated CSV summary:
  user_data/bulk_results/summary.csv

Usage example:
  source .venv/bin/activate
  python scripts/bulk_backtest.py --timerange -30d --timeframes 5m 15m 30m 1h --limit-pairs 40

Be cautious: Running many combinations can take a long time.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import signal
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
USER_DATA = ROOT / "user_data"
DATA_DIR = USER_DATA / "data" / "binance"
RESULT_DIR = USER_DATA / "bulk_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
STRATEGY_DIR = USER_DATA / "strategies"

# Global flag for graceful shutdown
interrupted = False


def signal_handler(signum, frame):
    global interrupted
    print(f"\n\nReceived interrupt signal ({signum}). Finishing current backtest then exiting...")
    print("Press Ctrl+C again to force quit immediately.")
    interrupted = True


def setup_signal_handlers():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def discover_strategy_classes() -> list[str]:
    """Recursively discover strategy class names (subclasses of IStrategy).

    Excludes some known placeholder / helper classes by name.
    """
    names: list[str] = []
    for py in STRATEGY_DIR.rglob("*.py"):
        # Skip obvious non-strategy helpers
        if py.name.startswith("_"):
            continue
        try:
            text = py.read_text(errors="ignore")
        except Exception:
            continue
        if "IStrategy" not in text:
            continue
        for line in text.splitlines():
            if "class" in line and "IStrategy" in line:
                m = re.search(r"class\s+([A-Za-z0-9_]+)\(IStrategy\)", line)
                if m:
                    cn = m.group(1)
                    if cn.lower() in {"doesnothingstrategy"}:  # filter noise
                        continue
                    names.append(cn)
    return sorted(set(names))


def discover_futures_strategy_names() -> set:
    """Return strategy class names in 'futures' subdirectory (skipped by default in spot)."""
    futdir = STRATEGY_DIR / "futures"
    if not futdir.exists():
        return set()
    names: set = set()
    for py in futdir.rglob("*.py"):
        try:
            text = py.read_text(errors="ignore")
        except Exception:
            continue
        if "IStrategy" not in text:
            continue
        for line in text.splitlines():
            if "class" in line and "IStrategy" in line:
                m = re.search(r"class\s+([A-Za-z0-9_]+)\(IStrategy\)", line)
                if m:
                    names.add(m.group(1))
    return names


def discover_pairs(timeframe: str) -> list[str]:
    pairs = []
    pattern = f"*-{timeframe}.feather"
    for f in DATA_DIR.glob(pattern):
        base = f.name.split(f"-{timeframe}")[0]
        if base.endswith("_USDT"):
            pairs.append(base.replace("_", "/"))
    return sorted(set(pairs))


def discover_pairs_with_volume(
    timeframe: str, lookback_days: int, top_n: int
) -> list[str]:
    """Return top_n pairs by average quote volume (close * volume) over lookback_days.

    Falls back to basic discovery if errors occur.
    Memory-optimized: processes files one by one and immediately releases memory.
    """
    base_pairs = discover_pairs(timeframe)
    if top_n <= 0 or not base_pairs:
        return base_pairs

    # Limit base_pairs to reduce memory usage
    if len(base_pairs) > 500:
        print(
            f"Warning: {len(base_pairs)} pairs found, limiting to first 500 for memory efficiency"
        )
        base_pairs = base_pairs[:500]

    minutes = 0
    try:
        if timeframe.endswith("m"):
            minutes = int(timeframe[:-1])
        elif timeframe.endswith("h"):
            minutes = int(timeframe[:-1]) * 60
        elif timeframe.endswith("d"):
            minutes = int(timeframe[:-1]) * 60 * 24
    except Exception:
        minutes = 0
    if minutes <= 0:
        return base_pairs

    # Limit lookback to prevent excessive memory usage
    max_candles = 2000  # ~7 days for 5m, ~83 days for 1h
    needed_candles = min(int((lookback_days * 24 * 60) / minutes), max_candles)

    scored: list[tuple[str, float]] = []
    print(f"Volume ranking {len(base_pairs)} pairs (using last {needed_candles} candles)...")

    for i, pair in enumerate(base_pairs):
        if i % 50 == 0:
            print(f"  Processing volume data: {i + 1}/{len(base_pairs)}")

        fname = pair.replace("/", "_") + f"-{timeframe}.feather"
        fpath = DATA_DIR / fname
        try:
            # Read only required columns to save memory
            df = pd.read_feather(fpath, columns=["close", "volume"])
            if df.empty:
                continue

            # Use tail efficiently and calculate immediately
            if len(df) > needed_candles:
                tail = df.iloc[-needed_candles:]
            else:
                tail = df

            if "close" not in tail.columns or "volume" not in tail.columns:
                continue

            quote_vol = (tail["close"] * tail["volume"]).mean()
            if pd.isna(quote_vol):
                continue

            scored.append((pair, float(quote_vol)))

            # Explicitly delete to free memory immediately
            del df, tail

        except Exception:
            continue

    if not scored:
        print("Warning: No pairs with valid volume data found, using basic discovery")
        return base_pairs

    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [p for p, _ in scored[:top_n]]
    print(f"Selected top {len(selected)} pairs by volume")
    return selected


def discover_timeframes() -> list[str]:
    """Infer available timeframes from downloaded data files."""
    tfs = set()
    for f in DATA_DIR.glob("*.feather"):
        name = f.name
        if "-" not in name:
            continue
        tf = name.rsplit("-", 1)[-1].replace(".feather", "")
        if not tf:
            continue
        tfs.add(tf)

    # Sort by approximate minutes
    def tf_key(tf: str) -> int:
        try:
            if tf.endswith("m"):
                return int(tf[:-1])
            if tf.endswith("h"):
                return int(tf[:-1]) * 60
            if tf.endswith("d"):
                return int(tf[:-1]) * 60 * 24
        except Exception:
            return 10**9
        return 10**9

    return sorted(tfs, key=tf_key)


def build_temp_config(
    base_config: Path, pairs: list[str], timeframe: str, stake_amount: float
) -> Path:
    cfg = json.loads(base_config.read_text())
    cfg["timeframe"] = timeframe
    cfg["exchange"]["pair_whitelist"] = pairs
    cfg["stake_amount"] = stake_amount
    cfg["max_open_trades"] = min(len(pairs), 100)
    # Do NOT add AgeFilter: Not supported for backtesting (causes failure).
    # Leave only StaticPairList.
    cfg["pairlists"] = [
        p for p in (cfg.get("pairlists") or []) if p.get("method") == "StaticPairList"
    ] or [{"method": "StaticPairList"}]
    tmp = Path(tempfile.mkstemp(prefix="btcfg_", suffix=".json")[1])
    tmp.write_text(json.dumps(cfg, indent=2))
    return tmp


def run_backtest(
    strategy: str,
    timeframe: str,
    pairs: list[str],
    timerange: str,
    base_config: Path,
    stake_amount: float,
) -> tuple[Path, bool]:
    """Run a single backtest and return (basestem, success).

    Handles negative timerange values by using the --timerange=<value> form so argparse
    inside freqtrade does not misinterpret e.g. "-60d" as another option.

    Freqtrade now produces a meta.json + zip when passing a filename without extension
    or a directory path. We'll standardize on a basestem (no extension) so we can look
    for either .json (legacy), .zip (new) or .log.
    """
    cfg = build_temp_config(base_config, pairs, timeframe, stake_amount)
    basestem = RESULT_DIR / f"{strategy}_{timeframe}"
    # Use basestem as backtest-filename root so freqtrade appends timestamp.
    # Support relative form -Nd by converting to explicit YYYYMMDD-YYYYMMDD (inclusive range)
    rel = re.fullmatch(r"-(\d+)d", timerange.strip())
    if rel:
        days = int(rel.group(1))
        end = datetime.utcnow().date()
        start = end - timedelta(days=days)
        timerange_val = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"
    else:
        timerange_val = timerange
    # Always pass timerange with '=' to be safe (supports values starting with '-')
    timerange_arg = f"--timerange={timerange_val}"
    # Determine interpreter (prefer local .venv python)
    venv_py = ROOT / ".venv" / "bin" / "python"
    py_exec = str(venv_py) if venv_py.exists() else sys.executable
    cmd = [
        py_exec,
        "-m",
        "freqtrade",
        "backtesting",
        "-c",
        str(cfg),
        "-s",
        strategy,
        "-i",
        timeframe,
        "--recursive-strategy-search",
        timerange_arg,
        "--export",
        "trades",
        "--backtest-filename",
        str(basestem),
    ]
    print("Running:", " ".join(cmd))
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
    except KeyboardInterrupt:
        print(f"\nBacktest for {strategy} {timeframe} was interrupted.")
        cfg.unlink(missing_ok=True)
        return basestem, False
    log_path = RESULT_DIR / f"{strategy}_{timeframe}.log"
    log_path.write_text(res.stdout + "\nSTDERR:\n" + res.stderr)
    if res.returncode != 0:
        print(f"Backtest failed for {strategy} {timeframe}")
    cfg.unlink(missing_ok=True)
    return basestem, res.returncode == 0


def _parse_new_format_zip(basestem: Path) -> dict[str, Any] | None:
    """Locate latest zip matching basestem pattern and return loaded main json (new format)."""
    zips = sorted(basestem.parent.glob(basestem.name + "-*.zip"))
    if not zips:
        return None
    z = zips[-1]
    try:
        with zipfile.ZipFile(z) as zf:
            # pick main json (no _config)
            candidates = [n for n in zf.namelist() if n.endswith(".json") and "_config" not in n]
            if not candidates:
                return None
            data = json.loads(zf.read(candidates[0]).decode())
            return data
    except Exception:
        return None


def _summarize_trades_list(
    trades: Iterable[dict[str, Any]], strategy: str, timeframe: str
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for t in trades:
        pair = t.get("pair")
        if not pair:
            continue
        r = rows.setdefault(
            pair,
            {
                "strategy": strategy,
                "timeframe": timeframe,
                "pair": pair,
                "trades": 0,
                "profit_abs": 0.0,
                "profit_pct_sum": 0.0,
                "wins": 0,
                "losses": 0,
                "profit_abs_win_sum": 0.0,
                "profit_abs_loss_sum": 0.0,
                "profit_pct_win_sum": 0.0,
                "profit_pct_loss_sum": 0.0,
            },
        )
        r["trades"] += 1
        pa = t.get("profit_abs") or t.get("close_profit_abs", 0.0) or 0.0
        pr = (
            t.get("profit_ratio") or t.get("close_profit_ratio", 0.0) or 0.0
        )  # ratio (e.g. 0.01 = 1%)
        r["profit_abs"] += pa
        r["profit_pct_sum"] += pr * 100.0
        if pa > 0:
            r["wins"] += 1
            r["profit_abs_win_sum"] += pa
            r["profit_pct_win_sum"] += pr * 100.0
        elif pa < 0:
            r["losses"] += 1
            r["profit_abs_loss_sum"] += pa  # negative
            r["profit_pct_loss_sum"] += pr * 100.0  # negative
    out: list[dict[str, Any]] = []
    for r in rows.values():
        tr = r["trades"]
        wins = r["wins"]
        losses = r["losses"]
        r["avg_profit_pct"] = r["profit_pct_sum"] / tr if tr else 0.0
        r["winrate_pct"] = (wins / tr * 100.0) if tr else 0.0
        r["avg_win_pct"] = (r["profit_pct_win_sum"] / wins) if wins else 0.0
        r["avg_loss_pct"] = (r["profit_pct_loss_sum"] / losses) if losses else 0.0
        # Profit factor: sum wins / abs(sum losses)
        r["profit_factor"] = (
            (r["profit_abs_win_sum"] / abs(r["profit_abs_loss_sum"]))
            if r["profit_abs_loss_sum"] < 0
            else (math.inf if r["profit_abs_win_sum"] > 0 else 0.0)
        )
        # Expectancy (percent) = winrate * avg_win + (1 - winrate)*avg_loss
        wr = r["winrate_pct"] / 100.0
        r["expectancy_pct"] = wr * r["avg_win_pct"] + (1 - wr) * r["avg_loss_pct"]
        r["avg_profit_abs_per_trade"] = r["profit_abs"] / tr if tr else 0.0
        out.append(r)
    return out


def _parse_log_pairs(log_path: Path, strategy: str, timeframe: str) -> list[dict[str, Any]]:
    """Fallback: Parse backtesting report table from log output if trade export empty."""
    if not log_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    inside = False
    for line in log_path.read_text(errors="ignore").splitlines():
        if "BACKTESTING REPORT" in line:
            inside = True
            continue
        if inside:
            if line.startswith("└") or line.startswith("+----"):
                inside = False
                continue
            if line.startswith("│") and "TOTAL" not in line and "Pair" not in line:
                # Split on │ and strip
                parts = [p.strip() for p in line.strip("│").split("│")]
                if len(parts) >= 5:
                    pair, trades, avg_pct, tot_profit_usdt = parts[0], parts[1], parts[2], parts[3]
                    try:
                        rows.append(
                            {
                                "strategy": strategy,
                                "timeframe": timeframe,
                                "pair": pair,
                                "trades": int(trades),
                                "profit_abs": float(tot_profit_usdt),
                                "avg_profit_pct": float(avg_pct),
                                "winrate_pct": None,
                                "wins": None,
                                "losses": None,
                            }
                        )
                    except Exception:
                        pass
    return rows


def summarize_results(basestem: Path, strategy: str, timeframe: str) -> list[dict[str, Any]]:
    """Attempt to summarize trades from (legacy json OR new zip format). Fallback to log parsing."""
    # Legacy single json path
    legacy_json = basestem.with_suffix(".json")
    if legacy_json.exists():
        try:
            data = json.loads(legacy_json.read_text())
            trades = data.get("trades") or []
            if trades:
                return _summarize_trades_list(trades, strategy, timeframe)
        except Exception:
            pass
    # New format: zip with nested structure
    data = _parse_new_format_zip(basestem)
    if data:
        # Try variant: data['strategy'][strategy]['trades']
        try:
            strat_node = data.get("strategy", {}).get(strategy)
            if strat_node and isinstance(strat_node, dict):
                for key in ("trades", "closed_trades"):
                    trades = strat_node.get(key)
                    if trades:
                        return _summarize_trades_list(trades, strategy, timeframe)
        except Exception:
            pass
    # Fallback: parse log table
    log_rows = _parse_log_pairs(
        basestem.parent / f"{strategy}_{timeframe}.log", strategy, timeframe
    )
    return log_rows


def write_summary(rows: list[dict[str, Any]], top_pairs: int):
    if not rows:
        print("No data to summarize.")
        return
    # Extended fieldnames for per-pair detail
    fieldnames = [
        "strategy",
        "timeframe",
        "pair",
        "trades",
        "profit_abs",
        "avg_profit_pct",
        "winrate_pct",
        "wins",
        "losses",
        "avg_win_pct",
        "avg_loss_pct",
        "profit_factor",
        "expectancy_pct",
        "avg_profit_abs_per_trade",
    ]
    rows_sorted = sorted(rows, key=lambda r: (r["strategy"], r["timeframe"], -r["profit_abs"]))
    with (RESULT_DIR / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_sorted:
            outrow = {k: r.get(k, "") for k in fieldnames}
            w.writerow(outrow)
    print("Summary written to", RESULT_DIR / "summary.csv")

    # Aggregated per (strategy,timeframe)
    agg: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        key = (r["strategy"], r["timeframe"])
        a = agg.setdefault(
            key,
            {
                "strategy": r["strategy"],
                "timeframe": r["timeframe"],
                "pairs": 0,
                "total_trades": 0,
                "total_profit_abs": 0.0,
                "weighted_avg_profit_pct": 0.0,
                "total_wins": 0,
                "total_losses": 0,
                "win_profit_abs_sum": 0.0,
                "loss_profit_abs_sum": 0.0,
                "win_profit_pct_sum": 0.0,
                "loss_profit_pct_sum": 0.0,
            },
        )
        tr = r.get("trades") or 0
        a["pairs"] += 1
        a["total_trades"] += tr
        pa = r.get("profit_abs") or 0.0
        a["total_profit_abs"] += pa
        a["total_wins"] += r.get("wins") or 0
        a["total_losses"] += r.get("losses") or 0
        a["weighted_avg_profit_pct"] += (r.get("avg_profit_pct") or 0.0) * tr
        # approximate win/loss splits from pair-level aggregated sums if present
        a["win_profit_abs_sum"] += r.get("profit_abs_win_sum", 0.0)
        a["loss_profit_abs_sum"] += r.get("profit_abs_loss_sum", 0.0)
        a["win_profit_pct_sum"] += r.get("profit_pct_win_sum", 0.0)
        a["loss_profit_pct_sum"] += r.get("profit_pct_loss_sum", 0.0)
    agg_rows: list[dict[str, Any]] = []
    for a in agg.values():
        tr = a["total_trades"]
        a["weighted_avg_profit_pct"] = a["weighted_avg_profit_pct"] / tr if tr else 0.0
        a["winrate_pct"] = (a["total_wins"] / tr * 100.0) if tr else 0.0
        # Derived aggregated metrics
        if a["total_wins"]:
            avg_win_pct = a["win_profit_pct_sum"] / a["total_wins"]
        else:
            avg_win_pct = 0.0
        if a["total_losses"]:
            avg_loss_pct = a["loss_profit_pct_sum"] / a["total_losses"]
        else:
            avg_loss_pct = 0.0
        if a["loss_profit_abs_sum"] < 0:
            profit_factor = (
                a["win_profit_abs_sum"] / abs(a["loss_profit_abs_sum"])
                if a["win_profit_abs_sum"]
                else 0.0
            )
        else:
            profit_factor = math.inf if a["win_profit_abs_sum"] > 0 else 0.0
        wr = a["winrate_pct"] / 100.0
        expectancy_pct = wr * avg_win_pct + (1 - wr) * avg_loss_pct
        a["profit_factor"] = profit_factor
        a["expectancy_pct"] = expectancy_pct
        a["avg_win_pct"] = avg_win_pct
        a["avg_loss_pct"] = avg_loss_pct
        a["avg_profit_abs_per_trade"] = a["total_profit_abs"] / tr if tr else 0.0
        agg_rows.append(a)
    # Ranking order: total_profit_abs desc, then profit_factor desc, then expectancy_pct desc
    agg_rows.sort(
        key=lambda x: (
            -x["total_profit_abs"],
            -x["profit_factor"] if math.isfinite(x["profit_factor"]) else float("-inf"),
            -x["expectancy_pct"],
        )
    )
    agg_fields = [
        "strategy",
        "timeframe",
        "pairs",
        "total_trades",
        "total_profit_abs",
        "weighted_avg_profit_pct",
        "winrate_pct",
        "profit_factor",
        "expectancy_pct",
        "avg_win_pct",
        "avg_loss_pct",
        "avg_profit_abs_per_trade",
        "total_wins",
        "total_losses",
    ]
    with (RESULT_DIR / "summary_aggregated.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        for a in agg_rows:
            clean = {k: a.get(k, "") for k in agg_fields}
            w.writerow(clean)
    print("Aggregated summary written to", RESULT_DIR / "summary_aggregated.csv")

    # Top pairs per (strategy,timeframe)
    if top_pairs > 0:
        top_rows: list[dict[str, Any]] = []
        group: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for r in rows:
            group.setdefault((r["strategy"], r["timeframe"]), []).append(r)
        for key, glist in group.items():
            glist_sorted = sorted(glist, key=lambda r: r["profit_abs"], reverse=True)[:top_pairs]
            for r in glist_sorted:
                top_rows.append(r)
        with (RESULT_DIR / "summary_top_pairs.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in top_rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        print("Top pairs file written to", RESULT_DIR / "summary_top_pairs.csv")

    # Overall best combination file (single best entry per strategy by best timeframe)
    best_combo_rows: list[dict[str, Any]] = []
    seen_strat: set = set()
    for a in agg_rows:
        if a["strategy"] in seen_strat:
            continue
        seen_strat.add(a["strategy"])
        best_combo_rows.append(a)
    with (RESULT_DIR / "best_combos.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        for a in best_combo_rows:
            clean = {k: a.get(k, "") for k in agg_fields}
            w.writerow(clean)
    print("Best combos written to", RESULT_DIR / "best_combos.csv")


def main():
    setup_signal_handlers()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--timerange", default="-30d", help="Timerange (e.g. -60d or 20240101-20240201)"
    )
    ap.add_argument(
        "--timeframes",
        nargs="*",
        default=None,
        help="Explicit list of timeframes. If omitted, auto-discover.",
    )
    ap.add_argument(
        "--limit-pairs",
        type=int,
        default=0,
        help="Limit number of pairs per timeframe (0 or negative = all).",
    )
    ap.add_argument(
        "--strategies", nargs="*", help="List of strategies to run (default: all discovered)."
    )
    ap.add_argument(
        "--pairs",
        nargs="*",
        help="Explicit pairs list (default: discover from data per timeframe).",
    )
    ap.add_argument("--stake-amount", type=float, default=100, help="Stake amount override.")
    ap.add_argument(
        "--top-pairs",
        type=int,
        default=5,
        help="Store top N pairs per strategy/timeframe (default 5, 0 to disable).",
    )
    ap.add_argument(
        "--volume-filter",
        type=int,
        default=0,
        help="Select top N pairs by average quote volume (close*volume). "
        "WARNING: High values use significant memory.",
    )
    ap.add_argument(
        "--volume-lookback-days",
        type=int,
        default=7,
        help="Lookback days for volume filter (default 7, max recommended: 14).",
    )
    ap.add_argument(
        "--include-futures",
        action="store_true",
        help="Include strategies in 'futures' subdirectory (default: skip).",
    )
    ap.add_argument(
        "--exclude-strategies",
        nargs="*",
        default=None,
        help="Explicit list of strategy names to exclude.",
    )
    args = ap.parse_args()

    # Normalize timeframes: allow comma-separated list (e.g. "5m,15m,30m,1h")
    if args.timeframes:
        original = list(args.timeframes)
        norm: list[str] = []
        for tf in args.timeframes:
            # Split on commas and whitespace
            parts = [p.strip() for p in re.split(r"[\s,]+", tf) if p.strip()]
            norm.extend(parts)
        # De-duplicate while preserving order
        seen = set()
        ordered: list[str] = []
        for tf in norm:
            if tf not in seen:
                seen.add(tf)
                ordered.append(tf)
        if ordered != original:
            print(f"Normalized timeframes input {original} -> {ordered}")
        args.timeframes = ordered

    strategies = args.strategies or discover_strategy_classes()
    futures_strats = discover_futures_strategy_names()
    if not args.include_futures:
        before = len(strategies)
        strategies = [s for s in strategies if s not in futures_strats]
        skipped = before - len(strategies)
        if skipped:
            print(f"Skipping {skipped} futures strategies (use --include-futures to include).")
    if args.exclude_strategies:
        excl = set(args.exclude_strategies)
        strategies = [s for s in strategies if s not in excl]
        print(f"Excluded strategies: {sorted(excl)}")
    if not strategies:
        print("No strategies discovered.")
        return 1
    print("Strategies:", strategies)

    base_config = USER_DATA / "config_backtest.json"
    all_rows: list[dict[str, Any]] = []
    timeframes = args.timeframes or discover_timeframes()
    print("Timeframes:", timeframes)
    total_combos = len(timeframes) * len(strategies)
    combo_index = 0

    try:
        # Execution order: Outer loop = timeframe, inner loop = strategy.
        for tf in timeframes:
            if interrupted:
                print("Interrupted. Stopping execution.")
                break

            if args.pairs:
                pairs = args.pairs
            elif args.volume_filter and args.volume_filter > 0:
                pairs = discover_pairs_with_volume(
                    tf, args.volume_lookback_days, args.volume_filter
                )
            else:
                pairs = discover_pairs(tf)
            original_count = len(pairs)
            if args.limit_pairs and args.limit_pairs > 0:
                pairs = pairs[: args.limit_pairs]
            volume_note = (
                f" volume-filter top {args.volume_filter}"
                if (args.volume_filter and args.volume_filter > 0 and not args.pairs)
                else ""
            )
            print(
                f"\n=== Timeframe {tf} start: {len(pairs)} pairs "
                f"(of {original_count}){volume_note} ==="
            )

            for strat in strategies:
                if interrupted:
                    print("Interrupted. Stopping execution.")
                    break

                combo_index += 1
                print(f"[Progress {combo_index}/{total_combos}] Running {strat} on {tf}...")
                basestem, ok = run_backtest(
                    strat, tf, pairs, args.timerange, base_config, args.stake_amount
                )
                if not ok:
                    # Skip summarizing failed runs (prevents empty / misleading rows)
                    print(f"  → FAILED: {strat} {tf}")
                    continue
                else:
                    print(f"  → SUCCESS: {strat} {tf}")
                all_rows.extend(summarize_results(basestem, strat, tf))

                # Force garbage collection after each strategy to free memory
                gc.collect()

    except KeyboardInterrupt:
        print("\nForce interrupted. Exiting immediately.")
        return 1

    if all_rows:
        write_summary(all_rows, args.top_pairs)
    else:
        print("No successful backtests to summarize.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
