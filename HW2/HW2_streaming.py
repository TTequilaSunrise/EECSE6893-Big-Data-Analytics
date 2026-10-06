"""HW2 Section 1: Simulated streaming analytics for AAPL on Spark (Dataproc).

Pulls 1-minute Massive/Polygon aggregates every 5 minutes over a 30-minute
wall-clock run (7 API calls), incrementally upserts into a Spark DataFrame,
and maintains a 30-minute moving-average DataFrame after each pull.

Do not put your API key in this file. Pass it as the first job argument
(or set MASSIVE_API_KEY). Prefer a completed past trading day — free-tier
current-day bars return 403.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# Both api.massive.com and api.polygon.io are valid (override with MASSIVE_BASE_URL).
DEFAULT_BASE_URL = "https://api.massive.com"
API_RETRIES = 5
API_RETRY_SLEEP_SEC = 10
TICKER = "AAPL"
NUM_PULLS = 7
PULL_INTERVAL_MIN = 5
WINDOW_MIN = 60
MA_WINDOW = 30

# Simulated stream start: completed past trading day, market hours (EDT 10:00 =
# 14:00 UTC). Advance this 1-hour window by 5 minutes on each pull.
STREAM_START_UTC = datetime(2026, 9, 25, 14, 0, 0, tzinfo=timezone.utc)

STOCK_SCHEMA = StructType(
    [
        StructField("Stock Name", StringType(), False),
        StructField("UTC Timestamp", TimestampType(), False),
        StructField("c", DoubleType(), True),
        StructField("l", DoubleType(), True),
        StructField("h", DoubleType(), True),
        StructField("o", DoubleType(), True),
        StructField("v", DoubleType(), True),
    ]
)

MA_SCHEMA = StructType(
    [
        StructField("Datetime", TimestampType(), False),
        StructField("c_MA", DoubleType(), True),
        StructField("l_MA", DoubleType(), True),
        StructField("h_MA", DoubleType(), True),
        StructField("o_MA", DoubleType(), True),
        StructField("v_MA", DoubleType(), True),
    ]
)


def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def api_base_url() -> str:
    return os.environ.get("MASSIVE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def get_histdata_massive(
    api_key: str,
    ticker: str,
    start_ms: int,
    end_ms: int,
    timespan: str = "minute",
    multiplier: int = 1,
) -> list[dict]:
    """Fetch aggregate bars. start/end are millisecond Unix timestamps."""
    base = api_base_url()
    path = (
        f"{base}/v2/aggs/ticker/{ticker}/range/"
        f"{multiplier}/{timespan}/{start_ms}/{end_ms}"
    )
    qs = urllib.parse.urlencode(
        {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": api_key,
        }
    )
    url = f"{path}?{qs}"
    last_err: Exception | None = None
    for attempt in range(1, API_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=90) as resp:
                payload = json.loads(resp.read().decode())
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            if exc.code == 403:
                raise RuntimeError(
                    "403 Forbidden: free tier blocks current-day realtime bars. "
                    "Use a completed past trading day."
                ) from exc
            raise RuntimeError(f"HTTP {exc.code}: {body[:500]}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last_err = exc
            print(
                f"  API attempt {attempt}/{API_RETRIES} failed ({exc}); "
                f"retry in {API_RETRY_SLEEP_SEC}s..."
            )
            if attempt < API_RETRIES:
                time.sleep(API_RETRY_SLEEP_SEC)
    else:
        raise RuntimeError(
            f"API unreachable after {API_RETRIES} attempts: {last_err}"
        ) from last_err

    results = payload.get("results") or []
    if not results:
        print(
            f"No results for window [{start_ms}, {end_ms}] "
            f"status={payload.get('status')}"
        )
    return results


def bars_to_rows(results: list[dict], ticker: str) -> list[tuple]:
    rows: list[tuple] = []
    for bar in results:
        ts = datetime.fromtimestamp(bar["t"] / 1000.0, tz=timezone.utc).replace(
            tzinfo=None
        )
        rows.append(
            (
                ticker,
                ts,
                float(bar["c"]),
                float(bar["l"]),
                float(bar["h"]),
                float(bar["o"]),
                float(bar["v"]),
            )
        )
    return rows


def incremental_upsert(existing: DataFrame | None, incoming: DataFrame) -> DataFrame:
    """Replace overlapping timestamps with the latest pull (anti-join + union)."""
    if existing is None:
        return incoming
    kept = existing.join(incoming, on="UTC Timestamp", how="left_anti")
    return kept.unionByName(incoming).orderBy("UTC Timestamp")


def compute_moving_averages(stock_df: DataFrame) -> DataFrame:
    """30-bar (≈30-minute) moving averages; Datetime is the window end."""
    ordered = stock_df.orderBy("UTC Timestamp")
    w = Window.orderBy("UTC Timestamp").rowsBetween(-(MA_WINDOW - 1), 0)
    ma = (
        ordered.withColumn("bar_count", F.count("c").over(w))
        .withColumn("c_MA", F.avg("c").over(w))
        .withColumn("l_MA", F.avg("l").over(w))
        .withColumn("h_MA", F.avg("h").over(w))
        .withColumn("o_MA", F.avg("o").over(w))
        .withColumn("v_MA", F.avg("v").over(w))
        .filter(F.col("bar_count") >= MA_WINDOW)
        .select(
            F.col("UTC Timestamp").alias("Datetime"),
            "c_MA",
            "l_MA",
            "h_MA",
            "o_MA",
            "v_MA",
        )
    )
    return ma


def incremental_ma_upsert(existing: DataFrame | None, incoming: DataFrame) -> DataFrame:
    if existing is None:
        return incoming
    kept = existing.join(incoming, on="Datetime", how="left_anti")
    return kept.unionByName(incoming).orderBy("Datetime")


def read_api_payload_from_gcs(spark: SparkSession, gcs_uri: str) -> dict:
    """Read a single JSON API response stored in GCS (one file per pull)."""
    lines = spark.read.text(gcs_uri).collect()
    body = "\n".join(row[0] for row in lines)
    return json.loads(body)


def load_pull_results(
    spark: SparkSession,
    api_key: str,
    start_ms: int,
    end_ms: int,
    pull_index: int,
    gcs_cache_uri: str | None,
) -> list[dict]:
    if gcs_cache_uri:
        uri = f"{gcs_cache_uri.rstrip('/')}/pull_{pull_index}.json"
        payload = read_api_payload_from_gcs(spark, uri)
        return payload.get("results") or []
    return get_histdata_massive(api_key, TICKER, start_ms, end_ms)


def parse_args(argv: list[str]) -> tuple[str, str, bool, str | None]:
    """Returns (api_key, output_uri, fast, gcs_cache_uri)."""
    args = list(argv)
    fast = False
    gcs_cache_uri: str | None = None
    if "--fast" in args:
        fast = True
        args.remove("--fast")
    if "--gcs-cache-uri" in args:
        idx = args.index("--gcs-cache-uri")
        if idx + 1 >= len(args):
            raise SystemExit("--gcs-cache-uri requires a gs:// path")
        gcs_cache_uri = args[idx + 1]
        del args[idx : idx + 2]

    api_key = os.environ.get("MASSIVE_API_KEY", "")
    output_uri = "gs://eecs-e6893-big-data-analysis-hw1/hw2/output/streaming"

    if args:
        api_key = args[0]
    if len(args) >= 2:
        output_uri = args[1]

    if gcs_cache_uri:
        if not api_key:
            api_key = "unused-with-gcs-cache"
    elif not api_key or api_key == "YOUR_API_KEY_HERE":
        raise SystemExit(
            "Pass API key as first arg or set MASSIVE_API_KEY. "
            "For Dataproc when API is blocked, prefetch locally and pass "
            "--gcs-cache-uri gs://.../hw2/api_pulls"
        )
    return api_key, output_uri, fast, gcs_cache_uri


def main() -> None:
    api_key, output_uri, fast, gcs_cache_uri = parse_args(sys.argv[1:])

    spark = (
        SparkSession.builder.appName("HW2-Streaming-AAPL")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    stock_df: DataFrame | None = None
    ma_df: DataFrame | None = None

    print("=" * 72)
    print("HW2 Section 1 — simulated AAPL 1-minute stream")
    print(f"stream start (UTC): {STREAM_START_UTC.isoformat()}")
    print(f"pulls: {NUM_PULLS} every {PULL_INTERVAL_MIN} min, window={WINDOW_MIN} min")
    print(f"fast mode (no sleep): {fast}")
    print(f"output: {output_uri}")
    print(f"api base: {api_base_url()}")
    print(f"gcs api cache: {gcs_cache_uri or '(live API)'}")
    print("=" * 72)

    for i in range(NUM_PULLS):
        offset = timedelta(minutes=i * PULL_INTERVAL_MIN)
        window_start = STREAM_START_UTC + offset
        window_end = window_start + timedelta(minutes=WINDOW_MIN)
        start_ms = to_ms(window_start)
        end_ms = to_ms(window_end)

        print(
            f"\n[pull {i + 1}/{NUM_PULLS}] "
            f"{window_start.isoformat()} -> {window_end.isoformat()}"
        )
        results = load_pull_results(
            spark, api_key, start_ms, end_ms, i, gcs_cache_uri
        )
        rows = bars_to_rows(results, TICKER)
        print(f"  bars returned: {len(rows)}")

        incoming = spark.createDataFrame(rows, schema=STOCK_SCHEMA)
        stock_df = incremental_upsert(stock_df, incoming).cache()
        stock_count = stock_df.count()
        print(f"  stock_df rows after upsert: {stock_count}")

        latest_ma = compute_moving_averages(stock_df)
        ma_df = incremental_ma_upsert(ma_df, latest_ma).cache()
        ma_count = ma_df.count()
        print(f"  ma_df rows after upsert: {ma_count}")

        print("  stock_df head:")
        stock_df.show(5, truncate=False)
        print("  ma_df tail:")
        ma_df.orderBy(F.col("Datetime").desc()).show(5, truncate=False)

        if i < NUM_PULLS - 1 and not fast:
            sleep_sec = PULL_INTERVAL_MIN * 60
            print(f"  sleeping {sleep_sec}s until next pull...")
            time.sleep(sleep_sec)

    assert stock_df is not None and ma_df is not None

    stock_path = f"{output_uri.rstrip('/')}/stock_bars"
    ma_path = f"{output_uri.rstrip('/')}/moving_averages"
    stock_df.coalesce(1).write.mode("overwrite").option("header", True).csv(stock_path)
    ma_df.coalesce(1).write.mode("overwrite").option("header", True).csv(ma_path)

    print("\n" + "=" * 72)
    print("FINAL stock_df")
    stock_df.orderBy("UTC Timestamp").show(20, truncate=False)
    print(f"stock_df count = {stock_df.count()}")
    print("FINAL ma_df")
    ma_df.orderBy("Datetime").show(20, truncate=False)
    print(f"ma_df count = {ma_df.count()}")
    print(f"Wrote stock bars -> {stock_path}")
    print(f"Wrote moving averages -> {ma_path}")
    print("=" * 72)

    spark.stop()


if __name__ == "__main__":
    main()
