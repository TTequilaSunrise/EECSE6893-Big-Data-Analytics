"""HW2 Section 3: Market Data Processing, EDA, Forecast & Backtest DAG.

Architecture:
  ingest_raw -> clean_validate -> feature_transform
       -> {eda_profile, risk_analytics, forecast_validate, simple_backtest}
       -> publish_report
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

from HW2_s3_eda_lib import (
    PipelinePaths,
    clean_validate,
    eda_profile,
    feature_transform,
    forecast_validate,
    ingest_raw,
    publish_report,
    risk_analytics,
    simple_backtest,
)

OUTPUT_DIR = Path.home() / "airflow" / "s3_eda_output"
PATHS = PipelinePaths(OUTPUT_DIR)


def _ingest(**_context: object) -> str:
    return ingest_raw(PATHS)


def _clean(**_context: object) -> str:
    return clean_validate(PATHS)


def _features(**_context: object) -> str:
    return feature_transform(PATHS)


def _eda(**_context: object) -> str:
    return eda_profile(PATHS)


def _risk(**_context: object) -> str:
    return risk_analytics(PATHS)


def _forecast(**_context: object) -> str:
    return forecast_validate(PATHS)


def _backtest(**_context: object) -> str:
    return simple_backtest(PATHS)


def _publish(**_context: object) -> str:
    return publish_report(PATHS)


default_args = {
    "owner": "yq2466",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="hw2_s3_market_eda",
    default_args=default_args,
    description="Section 3: cleaning + EDA + forecast + MA20 backtest",
    start_date=datetime(2026, 9, 29),
    schedule_interval=None,
    catchup=False,
    tags=["hw2", "section3", "eda", "backtest"],
) as dag:
    t_ingest = PythonOperator(task_id="ingest_raw", python_callable=_ingest)
    t_clean = PythonOperator(task_id="clean_validate", python_callable=_clean)
    t_feat = PythonOperator(task_id="feature_transform", python_callable=_features)
    t_eda = PythonOperator(task_id="eda_profile", python_callable=_eda)
    t_risk = PythonOperator(task_id="risk_analytics", python_callable=_risk)
    t_fc = PythonOperator(task_id="forecast_validate", python_callable=_forecast)
    t_bt = PythonOperator(task_id="simple_backtest", python_callable=_backtest)
    t_pub = PythonOperator(task_id="publish_report", python_callable=_publish)

    t_ingest >> t_clean >> t_feat >> [t_eda, t_risk, t_fc, t_bt] >> t_pub
