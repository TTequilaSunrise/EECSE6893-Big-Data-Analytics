"""HW2 Q2.2: End-to-end stock watchlist + high-volatility alert DAG.

Three sequential steps:
1) Fetch last 30 trading days via yfinance for 3 tickers
2) Compute 30-day return and average daily swing; write watchlist_summary.csv
3) Alert if latest-day swing exceeds threshold
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from airflow import DAG
from airflow.operators.python import PythonOperator

# Choose any 3 tickers available on yfinance
TICKERS = ["NVDA", "TSLA", "SPY"]
ALERT_THRESHOLD_PCT = 3.0  # latest-day swing >= 3% triggers ALERT
OUTPUT_DIR = Path.home() / "airflow" / "watchlist_output"
SUMMARY_CSV = OUTPUT_DIR / "watchlist_summary.csv"
PRICES_CSV = OUTPUT_DIR / "watchlist_prices.csv"


def fetch_data(**_context: object) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Pull a bit more than 30 calendar days so we have >= 30 trading days
    end = datetime.utcnow().date()
    start = end - timedelta(days=60)
    frames: list[pd.DataFrame] = []
    for ticker in TICKERS:
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            raise RuntimeError(f"No data returned for {ticker}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.reset_index()
        # Keep last 30 trading rows
        df = df.tail(30).copy()
        df["Ticker"] = ticker
        frames.append(df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]])
    prices = pd.concat(frames, ignore_index=True)
    prices.to_csv(PRICES_CSV, index=False)
    print(f"Fetched prices -> {PRICES_CSV} ({len(prices)} rows)")
    return str(PRICES_CSV)


def analytics_summary(**_context: object) -> str:
    prices = pd.read_csv(PRICES_CSV, parse_dates=["Date"])
    rows: list[dict[str, object]] = []
    for ticker in TICKERS:
        tdf = prices[prices["Ticker"] == ticker].sort_values("Date")
        if len(tdf) < 2:
            raise RuntimeError(f"Not enough rows for {ticker}")
        initial_close = float(tdf.iloc[0]["Close"])
        latest_close = float(tdf.iloc[-1]["Close"])
        ret_pct = (latest_close - initial_close) / initial_close * 100.0
        swing = ((tdf["High"] - tdf["Low"]) / tdf["Open"] * 100.0).mean()
        latest_swing = float(
            (tdf.iloc[-1]["High"] - tdf.iloc[-1]["Low"]) / tdf.iloc[-1]["Open"] * 100.0
        )
        latest_date = pd.Timestamp(tdf.iloc[-1]["Date"]).date().isoformat()
        rows.append(
            {
                "Ticker": ticker,
                "N_trading_days": int(len(tdf)),
                "Initial_Close": round(initial_close, 4),
                "Latest_Close": round(latest_close, 4),
                "Return_30d_pct": round(ret_pct, 4),
                "Avg_Daily_Swing_pct": round(float(swing), 4),
                "Latest_Date": latest_date,
                "Latest_Day_Swing_pct": round(latest_swing, 4),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    print("watchlist_summary.csv:")
    print(summary.to_string(index=False))
    print(f"Wrote -> {SUMMARY_CSV}")
    return str(SUMMARY_CSV)


def alert_notification(**_context: object) -> None:
    summary = pd.read_csv(SUMMARY_CSV)
    alerts: list[str] = []
    for _, row in summary.iterrows():
        swing = float(row["Latest_Day_Swing_pct"])
        if swing >= ALERT_THRESHOLD_PCT:
            msg = (
                f"ALERT: {row['Ticker']} experienced high volatility "
                f"({swing:.2f}% swing) on {row['Latest_Date']}"
            )
            alerts.append(msg)
            print(msg)
    if not alerts:
        print("All assets traded within normal volatility ranges.")
    else:
        print(f"Total alerts: {len(alerts)} (threshold={ALERT_THRESHOLD_PCT}%)")


default_args = {
    "owner": "yq2466",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="hw2_q22_watchlist",
    default_args=default_args,
    description="HW2 Q2.2 watchlist fetch/analytics/alert",
    start_date=datetime(2026, 9, 29),
    schedule_interval=None,  # manual trigger for screenshots
    catchup=False,
    tags=["hw2", "q2.2", "watchlist"],
) as dag:
    fetch = PythonOperator(task_id="fetch_data", python_callable=fetch_data)
    analytics = PythonOperator(
        task_id="analytics_summary", python_callable=analytics_summary
    )
    alert = PythonOperator(
        task_id="alert_notification", python_callable=alert_notification
    )
    fetch >> analytics >> alert
