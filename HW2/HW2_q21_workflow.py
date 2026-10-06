"""HW2 Q2.1: 19-task DAG matching the homework graph.

PythonOperator (yellow): t4, t7, t8, t11, t12, t15, t16, t19
BashOperator (grey):     t1, t2, t3, t5, t6, t9, t10, t13, t14, t17, t18

At least 3 different command styles per operator type.
"""

import time
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# --- Python callables (3+ different styles: print / sleep / count) ---


def py_print_message(task_id: str) -> None:
    print(f"[{task_id}] Hello from PythonOperator print style")


def py_sleep_brief(task_id: str, seconds: int = 2) -> None:
    print(f"[{task_id}] sleeping {seconds}s ...")
    time.sleep(seconds)
    print(f"[{task_id}] done sleeping")


def py_count_range(task_id: str, n: int = 5) -> int:
    total = sum(range(1, n + 1))
    print(f"[{task_id}] count 1..{n} = {total}")
    return total


default_args = {
    "owner": "yq2466",
    "retries": 0,
}

# First scheduled run as soon as possible: start_date a short time in the past,
# then every 30 minutes (schedule_interval). catchup=False avoids backfills.
with DAG(
    dag_id="hw2_q21_workflow",
    default_args=default_args,
    description="HW2 Q2.1 nineteen-task Python/Bash DAG",
    start_date=datetime(2026, 9, 30, 14, 0, 0),
    schedule_interval="*/30 * * * *",
    catchup=False,
    tags=["hw2", "q2.1"],
) as dag:
    # Bash styles: echo / bash -c script / python -c (run python one-liner via bash)
    t1 = BashOperator(task_id="t1", bash_command='echo "t1: bash echo start"')
    t2 = BashOperator(
        task_id="t2",
        bash_command='bash -c \'echo "t2: bash script"; date\'',
    )
    t3 = BashOperator(
        task_id="t3",
        bash_command='python3 -c \'print("t3: run python via bash")\'',
    )
    t5 = BashOperator(task_id="t5", bash_command='echo "t5: branch to t8/t9"')
    t6 = BashOperator(task_id="t6", bash_command='bash -c \'echo "t6: leaf"; sleep 1\'')
    t9 = BashOperator(
        task_id="t9",
        bash_command='python3 -c \'print("t9: python one-liner")\'',
    )
    t10 = BashOperator(task_id="t10", bash_command='echo "t10: toward t14"')
    t13 = BashOperator(
        task_id="t13",
        bash_command='bash -c \'echo "t13: toward t18"; true\'',
    )
    t14 = BashOperator(task_id="t14", bash_command='echo "t14: join hub"')
    t17 = BashOperator(
        task_id="t17",
        bash_command='python3 -c \'print("t17: bash->python")\'',
    )
    t18 = BashOperator(task_id="t18", bash_command='echo "t18: final bash join"')

    # Python styles: print / sleep / count
    t4 = PythonOperator(
        task_id="t4",
        python_callable=py_print_message,
        op_kwargs={"task_id": "t4"},
    )
    t7 = PythonOperator(
        task_id="t7",
        python_callable=py_sleep_brief,
        op_kwargs={"task_id": "t7", "seconds": 2},
    )
    t8 = PythonOperator(
        task_id="t8",
        python_callable=py_count_range,
        op_kwargs={"task_id": "t8", "n": 10},
    )
    t11 = PythonOperator(
        task_id="t11",
        python_callable=py_print_message,
        op_kwargs={"task_id": "t11"},
    )
    t12 = PythonOperator(
        task_id="t12",
        python_callable=py_sleep_brief,
        op_kwargs={"task_id": "t12", "seconds": 1},
    )
    t15 = PythonOperator(
        task_id="t15",
        python_callable=py_count_range,
        op_kwargs={"task_id": "t15", "n": 3},
    )
    t16 = PythonOperator(
        task_id="t16",
        python_callable=py_print_message,
        op_kwargs={"task_id": "t16"},
    )
    t19 = PythonOperator(
        task_id="t19",
        python_callable=py_count_range,
        op_kwargs={"task_id": "t19", "n": 19},
    )

    # Dependencies (match homework figure)
    t1 >> [t2, t3, t4, t5]
    t2 >> t6
    t3 >> [t7, t12]
    t5 >> [t8, t9]
    t8 >> [t10, t11, t12]
    t9 >> [t11, t12]
    t7 >> [t13, t14, t18]
    t10 >> t14
    t11 >> t14
    t12 >> [t14, t15]
    t13 >> t18
    t15 >> t18
    t14 >> [t16, t17]
    t17 >> t18
    t16 >> t19
    t18 >> t19
