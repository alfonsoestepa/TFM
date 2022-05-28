from lib.connect import connect_to_spark, load_data_parquet, load_data_csv
from lib.models import train_random_forrest, train_linear_svc, train_logistic_regression
from train import do_train_and_store_info_in_ml_flow

GLOBAL_SEED = 4321

if __name__ == "__main__":
    spark = connect_to_spark()
    orig_dataset = load_data_parquet(spark, "fraud2.parquet")
    new_dataset = load_data_csv(spark,
                                "/Users/alfonsoestepa/Documents/Master/TFM/PaySim/outputs/PS_20220528140235_176947716/PS_20220528140235_176947716_rawLog.csv")

    dataset = orig_dataset. \
        union(new_dataset)

    do_train_and_store_info_in_ml_flow(dataset, train_random_forrest, model_params={"seed": GLOBAL_SEED})
    do_train_and_store_info_in_ml_flow(dataset, train_logistic_regression, model_params={"seed": GLOBAL_SEED})
    do_train_and_store_info_in_ml_flow(dataset, train_linear_svc, model_params={"seed": GLOBAL_SEED})
