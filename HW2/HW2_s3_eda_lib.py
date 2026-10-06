"""HW2 Section 3: Market Data Processing, EDA & Forecast Pipeline.

Stages:
  ingest_raw -> clean_validate -> feature_transform
       -> {eda_profile, risk_analytics, forecast_validate} -> publish_report

Combines data-quality / exploratory analysis with walk-forward next-day High prediction.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = ["AAPL", "MSFT", "NVDA", "META", "AMZN"]
BENCHMARK = "SPY"
TICKERS = ASSETS + [BENCHMARK]

LOOKBACK_CALENDAR_DAYS = 180
ROLL_WINDOW = 20
REPORT_WINDOW = 30
FEATURE_WINDOW = 60
FORECAST_EVAL_DAYS = 5
VOL_Z_ALERT = 1.5


class PipelinePaths:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.raw_csv = output_dir / "raw_prices.csv"
        self.clean_csv = output_dir / "clean_prices.csv"
        self.qc_csv = output_dir / "data_quality_report.csv"
        self.features_csv = output_dir / "features.csv"
        self.desc_csv = output_dir / "eda_descriptive_stats.csv"
        self.corr_csv = output_dir / "eda_return_correlation.csv"
        self.weekday_csv = output_dir / "eda_weekday_returns.csv"
        self.risk_csv = output_dir / "risk_summary.csv"
        self.forecast_csv = output_dir / "forecast_errors.csv"
        self.backtest_csv = output_dir / "backtest_summary.csv"
        self.equity_csv = output_dir / "backtest_equity.csv"
        self.summary_txt = output_dir / "pipeline_summary.txt"


def download_daily_ohlcv(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    params = {
        "period1": int(
            datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp()
        ),
        "period2": int(
            datetime(end.year, end.month, end.day, tzinfo=timezone.utc).timestamp()
            + 24 * 3600
        ),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}"
        f"?{urllib.parse.urlencode(params)}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        return pd.DataFrame()
    block = result[0]
    timestamps = block.get("timestamp") or []
    quote = ((block.get("indicators") or {}).get("quote") or [{}])[0]
    adj = ((block.get("indicators") or {}).get("adjclose") or [{}])[0]
    adj_close = adj.get("adjclose") or quote.get("close")
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(timestamps, unit="s").tz_localize(None),
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
            "AdjClose": adj_close,
            "Volume": quote.get("volume"),
        }
    ).dropna(how="all")


def ols_beta(y: np.ndarray, x: np.ndarray) -> float:
    if len(y) < 5:
        return float("nan")
    x_design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    return float(coef[1])


def ols_predict(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray
) -> np.ndarray:
    x_design = np.column_stack([np.ones(len(train_x)), train_x])
    coef, *_ = np.linalg.lstsq(x_design, train_y, rcond=None)
    test_design = np.column_stack([np.ones(len(test_x)), test_x])
    return test_design @ coef


def ingest_raw(paths: PipelinePaths) -> str:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    end = datetime.utcnow().date()
    start = end - timedelta(days=LOOKBACK_CALENDAR_DAYS)
    frames: list[pd.DataFrame] = []
    for symbol in TICKERS:
        df = download_daily_ohlcv(
            symbol,
            datetime.combine(start, datetime.min.time()),
            datetime.combine(end, datetime.min.time()),
        )
        if df.empty:
            raise RuntimeError(f"No data returned for {symbol}")
        df["Ticker"] = symbol
        frames.append(
            df[["Date", "Ticker", "Open", "High", "Low", "Close", "AdjClose", "Volume"]]
        )
    raw = pd.concat(frames, ignore_index=True)
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.tz_localize(None)
    raw = raw.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    raw.to_csv(paths.raw_csv, index=False)
    print(f"Ingested raw rows={len(raw)} -> {paths.raw_csv}")
    return str(paths.raw_csv)


def clean_validate(paths: PipelinePaths) -> str:
    """Validate unadjusted OHLC; keep AdjClose for returns/features."""
    raw = pd.read_csv(paths.raw_csv, parse_dates=["Date"])
    df = raw.copy()
    for col in ["Open", "High", "Low", "Close", "AdjClose", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(
        subset=["Date", "Ticker", "Open", "High", "Low", "Close", "AdjClose", "Volume"]
    )
    df = df[df["Volume"] > 0]
    df = df[
        (df["High"] >= df["Low"])
        & (df["High"] >= df["Close"])
        & (df["Low"] <= df["Close"])
        & (df["High"] >= df["Open"])
        & (df["Low"] <= df["Open"])
    ]
    df = df.drop_duplicates(subset=["Ticker", "Date"], keep="last")
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    dropped = before - len(df)

    qc_rows: list[dict[str, object]] = []
    for ticker, tdf in df.groupby("Ticker"):
        tdf = tdf.sort_values("Date")
        gaps = tdf["Date"].diff().dt.days
        long_gaps = int((gaps > 3).sum())
        missing_frac_raw = float(
            raw.loc[
                raw["Ticker"] == ticker,
                ["Open", "High", "Low", "Close", "AdjClose", "Volume"],
            ]
            .isna()
            .mean()
            .mean()
        )
        ret = tdf["AdjClose"].pct_change()
        qc_rows.append(
            {
                "Ticker": ticker,
                "N_rows_clean": int(len(tdf)),
                "Date_start": tdf["Date"].min().date().isoformat(),
                "Date_end": tdf["Date"].max().date().isoformat(),
                "Missing_frac_raw": round(missing_frac_raw, 6),
                "Long_calendar_gaps": long_gaps,
                "Outlier_days_gt8pct": int((ret.abs() > 0.08).sum()),
                "Rows_dropped_global": dropped,
            }
        )

    df.to_csv(paths.clean_csv, index=False)
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(paths.qc_csv, index=False)
    print(qc.to_string(index=False))
    print(f"Cleaned -> {paths.clean_csv} (dropped {dropped})")
    return str(paths.clean_csv)


def feature_transform(paths: PipelinePaths) -> str:
    clean = pd.read_csv(paths.clean_csv, parse_dates=["Date"])
    parts: list[pd.DataFrame] = []
    for ticker, tdf in clean.groupby("Ticker"):
        tdf = tdf.sort_values("Date").copy()
        tdf["ret_1d"] = tdf["AdjClose"].pct_change()
        tdf["log_ret_1d"] = np.log(tdf["AdjClose"] / tdf["AdjClose"].shift(1))
        tdf["range_pct"] = (tdf["High"] - tdf["Low"]) / tdf["Open"] * 100.0
        tdf["log_volume"] = np.log1p(tdf["Volume"])
        tdf["vol_roll_20"] = tdf["ret_1d"].rolling(ROLL_WINDOW).std() * np.sqrt(252.0)
        vol_mean = tdf["vol_roll_20"].rolling(FEATURE_WINDOW).mean()
        vol_std = tdf["vol_roll_20"].rolling(FEATURE_WINDOW).std()
        tdf["vol_z"] = (tdf["vol_roll_20"] - vol_mean) / vol_std.replace(0, np.nan)
        tdf["ma_20"] = tdf["AdjClose"].rolling(ROLL_WINDOW).mean()
        tdf["dist_ma20_pct"] = (tdf["AdjClose"] / tdf["ma_20"] - 1.0) * 100.0
        tdf["next_high"] = tdf["High"].shift(-1)
        tdf["weekday"] = tdf["Date"].dt.day_name()
        tdf["Ticker"] = ticker
        parts.append(tdf)
    features = pd.concat(parts, ignore_index=True)
    features.to_csv(paths.features_csv, index=False)
    print(f"Features -> {paths.features_csv} ({len(features)} rows)")
    return str(paths.features_csv)


def eda_profile(paths: PipelinePaths) -> str:
    features = pd.read_csv(paths.features_csv, parse_dates=["Date"])
    metric_cols = ["ret_1d", "range_pct", "log_volume", "vol_roll_20", "dist_ma20_pct"]
    desc_rows: list[dict[str, object]] = []
    for ticker, tdf in features.groupby("Ticker"):
        for col in metric_cols:
            s = tdf[col].dropna()
            if s.empty:
                continue
            desc_rows.append(
                {
                    "Ticker": ticker,
                    "Metric": col,
                    "N": int(s.count()),
                    "Mean": round(float(s.mean()), 6),
                    "Std": round(float(s.std()), 6),
                    "P25": round(float(s.quantile(0.25)), 6),
                    "P50": round(float(s.quantile(0.50)), 6),
                    "P75": round(float(s.quantile(0.75)), 6),
                    "Min": round(float(s.min()), 6),
                    "Max": round(float(s.max()), 6),
                }
            )
    desc = pd.DataFrame(desc_rows)
    desc.to_csv(paths.desc_csv, index=False)

    pivot = (
        features.dropna(subset=["ret_1d"])
        .pivot_table(index="Date", columns="Ticker", values="ret_1d", aggfunc="last")
        .dropna(how="any")
    )
    corr = pivot.corr()
    corr.to_csv(paths.corr_csv)

    wd = (
        features.dropna(subset=["ret_1d"])
        .groupby(["Ticker", "weekday"], as_index=False)["ret_1d"]
        .mean()
    )
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    wd["weekday"] = pd.Categorical(wd["weekday"], categories=order, ordered=True)
    wd = wd.sort_values(["Ticker", "weekday"])
    wd["ret_1d"] = wd["ret_1d"].round(6)
    wd.to_csv(paths.weekday_csv, index=False)

    print("EDA descriptive preview:")
    print(desc.head(10).to_string(index=False))
    print("\nReturn correlation:")
    print(corr.round(3).to_string())
    return str(paths.desc_csv)


def risk_analytics(paths: PipelinePaths) -> str:
    features = pd.read_csv(paths.features_csv, parse_dates=["Date"])
    spy = features[features["Ticker"] == BENCHMARK].sort_values("Date")
    spy_ret = spy.set_index("Date")["ret_1d"]

    rows: list[dict[str, object]] = []
    for ticker in ASSETS:
        tdf = features[features["Ticker"] == ticker].sort_values("Date").dropna(
            subset=["ret_1d"]
        )
        aligned = tdf.set_index("Date")[
            ["AdjClose", "ret_1d", "vol_roll_20", "vol_z"]
        ].join(spy_ret.rename("spy_ret"), how="inner").dropna()
        if len(aligned) < REPORT_WINDOW + 5:
            raise RuntimeError(f"Insufficient history for {ticker}")

        window = aligned.tail(REPORT_WINDOW)
        initial = float(window.iloc[0]["AdjClose"])
        latest = float(window.iloc[-1]["AdjClose"])
        ret_30d = (latest - initial) / initial * 100.0

        spy_px = spy.set_index("Date").loc[window.index, "AdjClose"]
        spy_ret_30d = (
            (float(spy_px.iloc[-1]) - float(spy_px.iloc[0])) / float(spy_px.iloc[0]) * 100.0
        )
        beta = ols_beta(window["ret_1d"].to_numpy(), window["spy_ret"].to_numpy())
        latest_vol = float(window.iloc[-1]["vol_roll_20"]) * 100.0
        latest_z = (
            float(window.iloc[-1]["vol_z"]) if pd.notna(window.iloc[-1]["vol_z"]) else 0.0
        )
        rows.append(
            {
                "Ticker": ticker,
                "AsOf": pd.Timestamp(window.index[-1]).date().isoformat(),
                "Return_30d_pct": round(ret_30d, 4),
                "SPY_Return_30d_pct": round(spy_ret_30d, 4),
                "Excess_vs_SPY_pct": round(ret_30d - spy_ret_30d, 4),
                "Beta_vs_SPY": round(beta, 4),
                "Vol_20d_ann_pct": round(latest_vol, 4),
                "Vol_ZScore": round(latest_z, 4),
                "Vol_Regime": "ELEVATED" if latest_z >= VOL_Z_ALERT else "NORMAL",
            }
        )

    risk = pd.DataFrame(rows)
    risk.to_csv(paths.risk_csv, index=False)
    print(risk.to_string(index=False))
    return str(paths.risk_csv)


def forecast_validate(paths: PipelinePaths) -> str:
    """Walk-forward OLS: predict next-day High from same-day OHLCV (+ AdjClose)."""
    features = pd.read_csv(paths.features_csv, parse_dates=["Date"])
    feature_cols = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
    rows: list[dict[str, object]] = []

    for ticker in ASSETS:
        tdf = features[features["Ticker"] == ticker].sort_values("Date").copy()
        tdf = tdf.dropna(subset=feature_cols + ["next_high"]).reset_index(drop=True)
        if len(tdf) < FEATURE_WINDOW + FORECAST_EVAL_DAYS + 2:
            raise RuntimeError(f"Not enough rows to forecast {ticker}")

        eval_start = len(tdf) - FORECAST_EVAL_DAYS - 1
        abs_errors: list[float] = []
        for i in range(eval_start, len(tdf) - 1):
            train = tdf.iloc[:i]
            if len(train) < FEATURE_WINDOW:
                continue
            pred = float(
                ols_predict(
                    train[feature_cols].to_numpy(dtype=float),
                    train["next_high"].to_numpy(dtype=float),
                    tdf.iloc[[i]][feature_cols].to_numpy(dtype=float),
                )[0]
            )
            actual = float(tdf.iloc[i]["next_high"])
            rel_err = (pred - actual) / actual
            abs_errors.append(abs(rel_err))
            rows.append(
                {
                    "Ticker": ticker,
                    "Train_AsOf": pd.Timestamp(tdf.iloc[i]["Date"]).date().isoformat(),
                    "Target_Date": pd.Timestamp(tdf.iloc[i + 1]["Date"]).date().isoformat(),
                    "Predicted_High": round(pred, 4),
                    "Actual_High": round(actual, 4),
                    "Rel_Error": round(rel_err, 6),
                }
            )
        mae = float(np.mean(abs_errors)) if abs_errors else float("nan")
        print(f"{ticker}: MAE(|rel_error|)={mae:.4%}")

    forecast = pd.DataFrame(rows)
    forecast.to_csv(paths.forecast_csv, index=False)
    print(f"Wrote -> {paths.forecast_csv}")
    return str(paths.forecast_csv)


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _perf_stats(rets: pd.Series) -> dict[str, float]:
    r = rets.dropna()
    if r.empty:
        return {"total_return_pct": 0.0, "ann_sharpe": 0.0, "max_drawdown_pct": 0.0}
    equity = (1.0 + r).cumprod()
    vol = float(r.std())
    sharpe = float(r.mean() / vol * np.sqrt(252.0)) if vol > 0 else 0.0
    return {
        "total_return_pct": round((float(equity.iloc[-1]) - 1.0) * 100.0, 4),
        "ann_sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(_max_drawdown(equity) * 100.0, 4),
    }


def simple_backtest(paths: PipelinePaths) -> str:
    """Lightweight MA20 trend backtest (no look-ahead: signal at t, earn ret at t+1).

    Long when AdjClose > MA20, else flat. Compare to buy-and-hold per ticker,
    plus an equal-weight portfolio of the five assets vs SPY buy-and-hold.
    """
    features = pd.read_csv(paths.features_csv, parse_dates=["Date"])
    summary_rows: list[dict[str, object]] = []
    equity_frames: list[pd.DataFrame] = []

    for ticker in ASSETS + [BENCHMARK]:
        tdf = features[features["Ticker"] == ticker].sort_values("Date").copy()
        tdf = tdf.dropna(subset=["ret_1d", "ma_20", "AdjClose"]).reset_index(drop=True)
        # Position known at close of day t uses MA computed with day-t close;
        # return earned is next day's ret_1d -> shift position by 1.
        tdf["signal"] = (tdf["AdjClose"] > tdf["ma_20"]).astype(float)
        tdf["strat_ret"] = tdf["signal"].shift(1).fillna(0.0) * tdf["ret_1d"]
        tdf["bh_ret"] = tdf["ret_1d"]

        strat = _perf_stats(tdf["strat_ret"])
        bh = _perf_stats(tdf["bh_ret"])
        summary_rows.append(
            {
                "Ticker": ticker,
                "Strategy": "MA20_trend",
                "Strat_Return_pct": strat["total_return_pct"],
                "Strat_Sharpe": strat["ann_sharpe"],
                "Strat_MaxDD_pct": strat["max_drawdown_pct"],
                "BH_Return_pct": bh["total_return_pct"],
                "BH_Sharpe": bh["ann_sharpe"],
                "BH_MaxDD_pct": bh["max_drawdown_pct"],
            }
        )
        eq = pd.DataFrame(
            {
                "Date": tdf["Date"],
                "Ticker": ticker,
                "strat_equity": (1.0 + tdf["strat_ret"]).cumprod(),
                "bh_equity": (1.0 + tdf["bh_ret"]).cumprod(),
            }
        )
        equity_frames.append(eq)

    # Equal-weight portfolio of ASSETS using same MA rule
    wide: list[pd.Series] = []
    for ticker in ASSETS:
        tdf = features[features["Ticker"] == ticker].sort_values("Date")
        tdf = tdf.dropna(subset=["ret_1d", "ma_20", "AdjClose"])
        sig = (tdf["AdjClose"] > tdf["ma_20"]).astype(float)
        strat_ret = sig.shift(1).fillna(0.0) * tdf["ret_1d"]
        wide.append(pd.Series(strat_ret.to_numpy(), index=pd.to_datetime(tdf["Date"]), name=ticker))
    port = pd.concat(wide, axis=1).dropna(how="any").mean(axis=1)
    port_stats = _perf_stats(port)

    spy = features[features["Ticker"] == BENCHMARK].sort_values("Date")
    spy = spy.dropna(subset=["ret_1d"]).set_index("Date")["ret_1d"]
    spy.index = pd.to_datetime(spy.index)
    common = port.index.intersection(spy.index)
    port = port.loc[common]
    spy_aligned = spy.loc[common]
    spy_stats = _perf_stats(spy_aligned)

    summary_rows.append(
        {
            "Ticker": "EW_PORTFOLIO",
            "Strategy": "MA20_trend",
            "Strat_Return_pct": port_stats["total_return_pct"],
            "Strat_Sharpe": port_stats["ann_sharpe"],
            "Strat_MaxDD_pct": port_stats["max_drawdown_pct"],
            "BH_Return_pct": spy_stats["total_return_pct"],
            "BH_Sharpe": spy_stats["ann_sharpe"],
            "BH_MaxDD_pct": spy_stats["max_drawdown_pct"],
        }
    )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(paths.backtest_csv, index=False)

    port_eq = pd.DataFrame(
        {
            "Date": port.index,
            "Ticker": "EW_PORTFOLIO",
            "strat_equity": (1.0 + port).cumprod().to_numpy(),
            "bh_equity": (1.0 + spy_aligned).cumprod().to_numpy(),
        }
    )
    equity = pd.concat(equity_frames + [port_eq], ignore_index=True)
    equity.to_csv(paths.equity_csv, index=False)

    print("backtest_summary.csv:")
    print(summary.to_string(index=False))
    return str(paths.backtest_csv)


def publish_report(paths: PipelinePaths) -> str:
    qc = pd.read_csv(paths.qc_csv)
    corr = pd.read_csv(paths.corr_csv, index_col=0)
    risk = pd.read_csv(paths.risk_csv)
    forecast = pd.read_csv(paths.forecast_csv)
    backtest = pd.read_csv(paths.backtest_csv)
    features = pd.read_csv(paths.features_csv, parse_dates=["Date"])

    lines: list[str] = []
    lines.append("=== Market Data Processing + EDA + Forecast + Backtest ===")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z")
    lines.append(f"Universe: {', '.join(ASSETS)} | Benchmark: {BENCHMARK}")
    lines.append("")
    lines.append("-- Data quality --")
    lines.append(qc.to_string(index=False))
    lines.append("")

    corr_vals = corr.copy()
    np.fill_diagonal(corr_vals.values, np.nan)
    flat = corr_vals.stack().dropna()
    if not flat.empty:
        hi = flat.idxmax()
        lo = flat.idxmin()
        lines.append(f"Highest return corr: {hi[0]}–{hi[1]} = {flat.max():.3f}")
        lines.append(f"Lowest return corr: {lo[0]}–{lo[1]} = {flat.min():.3f}")

    lines.append("")
    lines.append("-- Risk snapshot (30d) --")
    lines.append(
        risk[
            ["Ticker", "Return_30d_pct", "Excess_vs_SPY_pct", "Beta_vs_SPY", "Vol_Regime"]
        ].to_string(index=False)
    )

    lines.append("")
    lines.append("-- Forecast (next-day High, last 5d MAE) --")
    mae = (
        forecast.assign(abs_err=lambda d: d["Rel_Error"].abs())
        .groupby("Ticker")["abs_err"]
        .mean()
        .sort_values()
    )
    for ticker, v in mae.items():
        lines.append(f"{ticker}: {v:.4%}")

    lines.append("")
    lines.append("-- MA20 trend backtest vs buy&hold --")
    lines.append(backtest.to_string(index=False))
    lines.append("")
    lines.append(f"Trading days (union): {features['Date'].nunique()}")

    text = "\n".join(lines) + "\n"
    paths.summary_txt.write_text(text, encoding="utf-8")
    print(text)
    return str(paths.summary_txt)


def run_pipeline(output_dir: Path) -> PipelinePaths:
    paths = PipelinePaths(output_dir)
    ingest_raw(paths)
    clean_validate(paths)
    feature_transform(paths)
    eda_profile(paths)
    risk_analytics(paths)
    forecast_validate(paths)
    simple_backtest(paths)
    publish_report(paths)
    return paths
