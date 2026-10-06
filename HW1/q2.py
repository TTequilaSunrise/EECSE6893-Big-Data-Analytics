"""HW1 Q2: Adult binary classification with Spark MLlib."""

import sys

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    LinearSVC,
    LogisticRegression,
    MultilayerPerceptronClassifier,
    NaiveBayes,
    OneVsRest,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BUCKET = "eecs-e6893-big-data-analysis-hw1"
DATA_PATH = f"gs://{BUCKET}/hw1/adult.data.csv"
OUTPUT_URI = f"gs://{BUCKET}/hw1/output/q2"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]
CAT = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]
NUM = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]


def load_adult(spark: SparkSession, path: str):
    df = spark.read.option("header", False).option("inferSchema", True).csv(path)
    for i, name in enumerate(COLUMNS):
        df = df.withColumnRenamed(f"_c{i}", name)
    for col in CAT + ["income"]:
        df = df.withColumn(col, F.trim(F.col(col)))
    df = df.dropna(how="any")
    for col in CAT:
        df = df.filter(F.col(col) != "?")
    return df


def preprocessor() -> Pipeline:
    stages: list[object] = [StringIndexer(inputCol="income", outputCol="label")]
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

    spark = SparkSession.builder.appName("HW1-Q2").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    raw = load_adult(spark, data_path)
    prepared = preprocessor().fit(raw).transform(raw).select("features", "label")
    train, test = prepared.randomSplit([0.7, 0.3], seed=100)
    train, test = train.cache(), test.cache()

    feat_dim = int(train.select("features").first()["features"].size)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    models = {
        "Logistic Regression": LogisticRegression(maxIter=50),
        "Random Forest": RandomForestClassifier(numTrees=50, seed=100),
        "Naive Bayes": NaiveBayes(),
        "Decision Tree": DecisionTreeClassifier(seed=100),
        "Gradient Boosting Trees": GBTClassifier(maxIter=50, seed=100),
        "Multilayer Perceptron": MultilayerPerceptronClassifier(
            layers=[feat_dim, 64, 32, 2], maxIter=100, seed=100, blockSize=128
        ),
        "Linear SVM": LinearSVC(maxIter=50),
        "One-vs-Rest": OneVsRest(classifier=LogisticRegression(maxIter=50)),
    }

    results: list[tuple[str, float]] = []
    for name, model in models.items():
        acc = float(evaluator.evaluate(model.fit(train).transform(test)))
        results.append((name, acc))
        print(f"{name}: {acc:.6f}")
    results.sort(key=lambda x: x[1], reverse=True)

    lines = ["model,accuracy"] + [f"{n},{a:.6f}" for n, a in results]
    spark.sparkContext.parallelize(lines, 1).saveAsTextFile(f"{output_uri}/accuracy_csv")
    spark.stop()


if __name__ == "__main__":
    main()
