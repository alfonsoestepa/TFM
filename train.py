from datetime import datetime

import mlflow.pyspark.ml

from lib.connect import connect_to_spark, load_data_csv
from lib.evaluation import evaluate_model, calculate_feature_importance
from lib.mlflow_utils import store_dataset_in_mlflow, store_confusion_matrix, store_model_in_mlflow, \
    store_feature_importances
from lib.models import train_logistic_regression, train_random_forrest, train_linear_svc
from lib.prepare_data import prepare_data


def do_train_and_store_info_in_ml_flow(dataset, train_model_function, seed, split_weights=[0.5, 0.5], model_params={}):
    start_time = datetime.now()
    with mlflow.start_run():
        preprocessed_data, train_data, test_data = prepare_data(dataset, seed, split_weights=split_weights)

        model, model_name, coefficients = train_model_function(preprocessed_data, **model_params)

        train_metrics, train_predictions = evaluate_model(model, train_data, prefix="train_")
        test_metrics, test_predictions = evaluate_model(model, test_data, prefix="test_")
        end_time = datetime.now()

        mlflow.set_tag("start_time", str(start_time))
        mlflow.set_tag("end_time", str(end_time))
        mlflow.set_tag("model", model_name)
        mlflow.log_metric("run_time", (end_time - start_time).total_seconds())
        mlflow.log_metric("dataset_size", preprocessed_data.count())

        mlflow.log_param("seed", seed)
        mlflow.log_param("train_size_percentage", split_weights[0])
        mlflow.log_param("test_size_percentage", split_weights[1])
        mlflow.log_metrics({**train_metrics, **test_metrics})
        store_model_in_mlflow(model, train_data, "fraud")
        store_dataset_in_mlflow(preprocessed_data)
        store_confusion_matrix(train_predictions, "train-confusion-matrix.png")
        store_confusion_matrix(test_predictions, "test-confusion-matrix.png")

        store_feature_importances(calculate_feature_importance(train_predictions, coefficients), "feature_importances.png")


if __name__ == "__main__":
    GLOBAL_SEED = 1234

    spark = connect_to_spark()
    raw_dataset = load_data_csv(spark, "data/train_test/fraud.csv")
    do_train_and_store_info_in_ml_flow(raw_dataset, train_logistic_regression, GLOBAL_SEED)
    do_train_and_store_info_in_ml_flow(raw_dataset, train_logistic_regression, GLOBAL_SEED, model_params={"regParam": 0.1})
    do_train_and_store_info_in_ml_flow(raw_dataset, train_logistic_regression, GLOBAL_SEED, model_params={"regParam": 0.01})
    do_train_and_store_info_in_ml_flow(raw_dataset, train_logistic_regression, GLOBAL_SEED, model_params={"regParam": 1})

    do_train_and_store_info_in_ml_flow(raw_dataset, train_random_forrest, GLOBAL_SEED, model_params={"seed": GLOBAL_SEED})
    do_train_and_store_info_in_ml_flow(raw_dataset, train_random_forrest, GLOBAL_SEED, model_params={"seed": GLOBAL_SEED, "numTrees": 5})
    do_train_and_store_info_in_ml_flow(raw_dataset, train_random_forrest, GLOBAL_SEED, model_params={"seed": GLOBAL_SEED, "numTrees": 50})
    do_train_and_store_info_in_ml_flow(raw_dataset, train_random_forrest, GLOBAL_SEED, model_params={"seed": GLOBAL_SEED, "numTrees": 100})

    do_train_and_store_info_in_ml_flow(raw_dataset, train_linear_svc, GLOBAL_SEED)
    do_train_and_store_info_in_ml_flow(raw_dataset, train_linear_svc, GLOBAL_SEED, model_params={"regParam": 0.1})
    do_train_and_store_info_in_ml_flow(raw_dataset, train_linear_svc, GLOBAL_SEED, model_params={"regParam": 0.01})
    do_train_and_store_info_in_ml_flow(raw_dataset, train_linear_svc, GLOBAL_SEED, model_params={"regParam": 1})

    print("Ok.")
