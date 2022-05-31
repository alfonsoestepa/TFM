import mlflow

from lib.connect import connect_to_spark, load_data_csv
from lib.evaluation import evaluate_model
from src.lib.mlflow_utils import store_confusion_matrix


if __name__ == "__main__":
    spark = connect_to_spark()
    validation_data = load_data_csv(spark, "data/validation/fraud.csv")
    #validation_data = load_data_csv(spark, "data/phase2/validation/fraud_future.csv")
    model_to_load = "models:/fraud/5"
    model = mlflow.spark.load_model(model_to_load)

    with mlflow.start_run():
        mlflow.set_tag("model", model_to_load)
        mlflow.log_metric("dataset_size", validation_data.count())
        evaluation_metrics, predictions = evaluate_model(model, validation_data, prefix="val_")
        mlflow.log_metrics(evaluation_metrics)
        store_confusion_matrix(predictions, "val_confusion_matrix.png")

    print("Ok.")
