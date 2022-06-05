from lib.connect import connect_to_spark, load_data_csv
from lib.models import train_random_forrest, train_linear_svc, train_logistic_regression
from train import do_train_and_store_info_in_ml_flow

if __name__ == "__main__":
    GLOBAL_SEED = 4321

    spark = connect_to_spark()
    orig_dataset = load_data_csv(spark, "data/train_test/fraud.csv")
    new_dataset = load_data_csv(spark, "data/phase2/train_test/fraud_future.csv")

    dataset = orig_dataset. \
        union(new_dataset)

    do_train_and_store_info_in_ml_flow(dataset, train_random_forrest, GLOBAL_SEED, model_params={"seed": GLOBAL_SEED})
    do_train_and_store_info_in_ml_flow(dataset, train_logistic_regression, GLOBAL_SEED)
    do_train_and_store_info_in_ml_flow(dataset, train_linear_svc, GLOBAL_SEED)

    print("Ok.")
