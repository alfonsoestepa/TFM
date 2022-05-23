import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import mlflow.pyspark.ml
import seaborn as sns
from mlflow.models.signature import infer_signature
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType

GLOBAL_SEED = 1234


def connect_to_spark():
    print("Connect to spark ...")
    conf = SparkConf()
    conf.set("spark.driver.memory", "14g")
    conf.set("spark.kryoserializer.buffer.max", "1024")
    conf.set("spark.jars.packages",
             "org.mlflow:mlflow-spark:1.25.1,org.apache.hadoop:hadoop-aws:3.3.1,com.amazonaws:aws-java-sdk-bundle:1.12.220")
    conf.set("fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    conf.set("fs.s3a.access.key", os.environ["AWS_ACCESS_KEY_ID"])
    conf.set("fs.s3a.secret.key", os.environ["AWS_SECRET_ACCESS_KEY"])
    conf.set("fs.s3a.endpoint", os.environ["MLFLOW_S3_ENDPOINT_URL"])
    conf.set("fs.s3a.path.style.access", "true")
    conf.set("fs.s3a.fast.upload", "true")
    conf.set("fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")

    sc = SparkContext.getOrCreate(conf)
    return SparkSession(sc)


def load_data(spark):
    print("Loading data ...")
    return spark. \
        read. \
        parquet("fraud2.parquet"). \
        withColumn("isFraud", col("isFraud").cast(DoubleType())). \
        withColumnRenamed("isFraud", "label")


def subsample_data(raw_dataset, seed=GLOBAL_SEED):
    fraud_op_count = raw_dataset[raw_dataset["label"] == 1].count()
    return raw_dataset.sampleBy("label", fractions={0: fraud_op_count / raw_dataset.count(), 1: 1}, seed=seed)


def prepare_data(raw_dataset, split_weights=[0.75, 0.25], seed=GLOBAL_SEED):
    print("Prepare data ...")
    preprocessed_data = subsample_data(raw_dataset, seed)
    tr_data, test_data = preprocessed_data.randomSplit(split_weights, seed=seed)
    return preprocessed_data, tr_data, test_data


def train_logistic_regression(train_data, **kwargs):
    print("Train model ...")
    name_encoder = StringIndexer(inputCols=["nameOrig", "nameDest", "type"],
                                 outputCols=["nameOrigIdx", "nameDestIdx", "typeIdx"], handleInvalid="keep")
    type_encoder = OneHotEncoder(inputCol="typeIdx", outputCol="typeEnc")
    assembler_tr = VectorAssembler(
        inputCols=["step", "amount", "oldbalanceOrg", "newbalanceOrig", "typeEnc", "nameOrigIdx", "nameDestIdx"],
        outputCol="features")
    model = LogisticRegression(**kwargs)

    pipeline_es = Pipeline(stages=[name_encoder, type_encoder, assembler_tr, model])
    return pipeline_es.fit(train_data), "logistic-regression"


def train_randon_forrest(train_data, **kwargs):
    print("Train model ...")
    name_encoder = StringIndexer(inputCols=["nameOrig", "nameDest", "type"],
                                 outputCols=["nameOrigIdx", "nameDestIdx", "typeIdx"], handleInvalid="keep")
    type_encoder = OneHotEncoder(inputCol="typeIdx", outputCol="typeEnc")
    assembler_tr = VectorAssembler(
        #inputCols=["step", "amount", "oldbalanceOrg", "oldbalanceDest", "newbalanceOrig", "newbalanceDest", "typeEnc"],
        inputCols=["amount", "oldbalanceOrg", "newbalanceOrig", "typeEnc"],
        outputCol="features")
    model = RandomForestClassifier(**kwargs)

    pipeline_es = Pipeline(stages=[name_encoder, type_encoder, assembler_tr, model])
    return pipeline_es.fit(train_data), "random-forrest"


def train_linear_svc(train_data, **kwargs):
    print("Train model ...")
    name_encoder = StringIndexer(inputCols=["nameOrig", "nameDest", "type"],
                                 outputCols=["nameOrigIdx", "nameDestIdx", "typeIdx"], handleInvalid="keep")
    type_encoder = OneHotEncoder(inputCol="typeIdx", outputCol="typeEnc")
    assembler_tr = VectorAssembler(
        inputCols=["step", "amount", "oldbalanceOrg", "oldbalanceDest", "newbalanceOrig", "newbalanceDest", "typeEnc"],
        outputCol="features")
    model = LinearSVC(**kwargs)

    pipeline_es = Pipeline(stages=[name_encoder, type_encoder, assembler_tr, model])
    return pipeline_es.fit(train_data), "linear-svc"


def store_dataset_in_mlflow(dataset):
    print("Storing data in mlflow ...")
    active_run = mlflow.active_run()
    temp_file = "fraud-{}.parquet".format(active_run.info.run_id)
    dataset.write.parquet(temp_file)
    mlflow.log_artifacts(temp_file, artifact_path="dataset")
    shutil.rmtree(temp_file)


def store_model_in_mlflow(model, train_data, model_name):
    print("Storing model in mlflow ...")

    params_to_store = {param[0].name: param[1] for param in model.stages[-1].extractParamMap().items()}
    mlflow.log_params(params_to_store)

    train_data_no_label = train_data.drop("label")

    signature = infer_signature(train_data_no_label, train_data.select("label"))

    return mlflow.spark.log_model(model, artifact_path="model", registered_model_name=model_name,
                                  signature=signature, dfs_tmpdir="s3://tmp",
                                  input_example=train_data_no_label.limit(5).toPandas())


def evaluate_model(model, test_data, prefix=""):
    print("Evaluating the model ...")
    pred_pipeline = model.transform(test_data)

    evaluator = BinaryClassificationEvaluator()
    roc = evaluator.evaluate(pred_pipeline)

    evaluator = MulticlassClassificationEvaluator()
    return {
        prefix + "accuracy": evaluator.evaluate(pred_pipeline, {evaluator.metricName: "accuracy"}),
        prefix + "f1": evaluator.evaluate(pred_pipeline),
        prefix + "weightedPrecision": evaluator.evaluate(pred_pipeline, {evaluator.metricName: "weightedPrecision"}),
        prefix + "weightedRecall": evaluator.evaluate(pred_pipeline, {evaluator.metricName: "weightedRecall"}),
        prefix + "AU-ROC": roc
    }


def store_confusion_matrix(model, data, file_name):
    print("Creating confusion matrix ...")
    pred_pipeline = model.transform(data)

    confusion_matrix = pred_pipeline.select(['prediction', 'label']).groupby("label").pivot(
        "prediction").count().drop("label").toPandas() / pred_pipeline.count()

    ax = sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='.2%')

    ax.set_title('Confusion Matrix\n\n');
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values');
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    plt.savefig(file_name)
    plt.clf()

    mlflow.log_artifact(file_name)
    os.remove(file_name)


def main():
    start_time = datetime.now()
    with mlflow.start_run():
        spark = connect_to_spark()
        raw_dataset = load_data(spark)
        split_weights = [0.5, 0.5]
        preprocessed_data, train_data, test_data = prepare_data(raw_dataset, split_weights=split_weights)
        #model, model_name = train_logistic_regression(preprocessed_data, maxIter=1000, regParam=0.01)
        #model, model_name = train_randon_forrest(preprocessed_data, numTrees=100, seed=GLOBAL_SEED)
        model, model_name = train_linear_svc(preprocessed_data, maxIter=1000, regParam=0.001)
        train_metrics = evaluate_model(model, train_data, prefix="train_")
        test_metrics = evaluate_model(model, test_data, prefix="test_")
        end_time = datetime.now()

        mlflow.set_tag("start_time", str(start_time))
        mlflow.set_tag("end_time", str(end_time))
        mlflow.set_tag("model", model_name)
        mlflow.log_metric("run_time", (end_time - start_time).total_seconds())
        mlflow.log_metric("dataset_size", preprocessed_data.count())

        mlflow.log_param("seed", GLOBAL_SEED)
        mlflow.log_param("train_size_percentage", split_weights[0])
        mlflow.log_param("test_size_percentage", split_weights[1])
        mlflow.log_metrics({**train_metrics, **test_metrics})
        store_model_in_mlflow(model, train_data, model_name)
        store_dataset_in_mlflow(preprocessed_data)
        store_confusion_matrix(model, train_data, "train-confusion-matrix.png")
        store_confusion_matrix(model, test_data, "test-confusion-matrix.png")


if __name__ == "__main__":
    main()
