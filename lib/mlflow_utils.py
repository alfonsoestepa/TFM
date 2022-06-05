import os
import shutil

import matplotlib.pyplot as plt
import mlflow.pyspark.ml
import seaborn as sns
from mlflow.models.signature import infer_signature

from .evaluation import calculate_confusion_matrix


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


def store_confusion_matrix(predictions, file_name):
    print("Creating confusion matrix ...")
    confusion_matrix = calculate_confusion_matrix(predictions)

    ax = sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='.2%')

    ax.set_title('Confusion Matrix\n\n');
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values');
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    plt.tight_layout()
    plt.savefig(file_name)
    plt.clf()

    mlflow.log_artifact(file_name)
    os.remove(file_name)


def store_feature_importances(feature_importances, file_name):
    print("Creating feature importances graph ...")
    fig, ax = plt.subplots()
    p1 = ax.bar(list(feature_importances.keys()), list(feature_importances.values()))
    ax.xaxis.set_tick_params(rotation=90)
    ax.bar_label(p1, label_type='center', rotation=90, fmt='%.2f')
    plt.tight_layout()
    plt.savefig(file_name)
    plt.clf()

    mlflow.log_artifact(file_name)
    os.remove(file_name)