"""HW1 Q3: Bank Marketing term-deposit prediction (Spark MLlib)."""

import sys

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    GBTClassifier,
    LinearSVC,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession

BUCKET = "eecs-e6893-big-data-analysis-hw1"
DATA_PATH = f"gs://{BUCKET}/hw1/bank-additional-full.csv"
OUTPUT_URI = f"gs://{BUCKET}/hw1/output/q3"

CAT = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]
NUM = [
    "age", "campaign", "pdays", "previous",
    "emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed",
]


def load_bank(spark: SparkSession, path: str):
    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("sep", ";")
        .csv(path)
    )
    for col in df.columns:
        if "." in col:
            df = df.withColumnRenamed(col, col.replace(".", "_"))
    return df.drop("duration").dropna(how="any")


def preprocessor() -> Pipeline:
    stages: list[object] = [StringIndexer(inputCol="y", outputCol="label")]
    for col in CAT:
        stages.append(StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCol=f"{col}_idx", outputCol=f"{col}_oh"))
    stages.append(
        VectorAssembler(inputCols=[f"{c}_oh" for c in CAT] + NUM, outputCol="features")
    )
    return Pipeline(stages=stages)


def main() -> None:
    args = sys.argv[1:]
    data_path, output_uri = args if len(args) == 2 else [DATA_PATH, OUTPUT_URI]

    spark = SparkSession.builder.appName("HW1-Q3").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    raw = load_bank(spark, data_path)
    prepared = preprocessor().fit(raw).transform(raw).select("features", "label")
    train, test = prepared.randomSplit([0.7, 0.3], seed=100)
    train, test = train.cache(), test.cache()

    acc_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    auc_eval = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    models = {
        "Logistic Regression": LogisticRegression(maxIter=50),
        "Random Forest": RandomForestClassifier(numTrees=50, seed=100),
        "Gradient Boosting Trees": GBTClassifier(maxIter=50, seed=100),
        "Linear SVM": LinearSVC(maxIter=50),
    }

    results: list[tuple[str, float, float]] = []
    for name, model in models.items():
        preds = model.fit(train).transform(test)
        acc, auc = float(acc_eval.evaluate(preds)), float(auc_eval.evaluate(preds))
        results.append((name, acc, auc))
        print(f"{name}: acc={acc:.6f}, AUC={auc:.6f}")
    results.sort(key=lambda t: (t[2], t[1]), reverse=True)

    lines = ["model,accuracy,auc"] + [f"{n},{a:.6f},{u:.6f}" for n, a, u in results]
    spark.sparkContext.parallelize(lines, 1).saveAsTextFile(f"{output_uri}/metrics_csv")
    spark.stop()


if __name__ == "__main__":
    main()
