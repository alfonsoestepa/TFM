import mlflow

from lib.connect import connect_to_spark, load_data_csv
from lib.evaluation import evaluate_model

if __name__ == "__main__":
    spark = connect_to_spark()
    test_data = load_data_csv(spark,
                      "/Users/alfonsoestepa/Documents/Master/TFM/PaySim/outputs/PS_20220528140153_176905439/PS_20220528140153_176905439_rawLog.csv")

    model = mlflow.spark.load_model("models:/random-forrest/8")

    print(evaluate_model(model, test_data))
