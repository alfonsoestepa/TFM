from pyspark.sql.functions import when, col, rand, lit

from lib.connect import connect_to_spark, load_data_csv

if __name__ == "__main__":
    spark = connect_to_spark()
    #data = load_data_csv(spark, "../PaySim/outputs/PS_20220529132440_261072141/PS_20220529132440_261072141_rawLog.csv")
    data = load_data_csv(spark, "data/phase2/validation/raw_fraud.csv")

    data.groupby("type", "label").count().show()

    data = data.withColumn("type", when(col("label") == 1.0,
                                        when(rand() < 0.85, lit("DEBIT")).otherwise(col("type")))
                           .otherwise(when(rand() < 0.55, lit("TRANSFER")).otherwise(col("type"))))

    data = data.withColumn("newbalanceOrig", when(col("label") == 1.0,
                                        when(rand() < 0.95, data.newbalanceOrig * (0.4 + rand())).otherwise(col("newbalanceOrig")))
                           .otherwise(when(rand() < 0.95, data.newbalanceOrig * (2.1 + rand())).otherwise(col("newbalanceOrig"))))

    data = data.withColumn("oldbalanceOrg", when(col("label") == 1.0,
                                        when(rand() < 0.75, data.oldbalanceOrg * (0.01 + rand())).otherwise(col("oldbalanceOrg")))
                           .otherwise(when(rand() < 0.95, data.oldbalanceOrg * (2.8 + rand())).otherwise(col("oldbalanceOrg"))))

    data = data.withColumn("amount", when(col("label") == 1.0,
                                        when(rand() < 0.75, data.amount * (0.01 + rand())).otherwise(col("amount")))
                           .otherwise(when(rand() < 0.95, data.amount * (1.9 + rand())).otherwise(col("amount"))))

    data.groupby("type", "label").count().show()

    data.withColumnRenamed("label", "isFraud").\
        withColumnRenamed("type", "action").\
        withColumnRenamed("oldbalanceOrg", "oldBalanceOrig").\
        withColumnRenamed("newbalanceOrig", "newBalanceOrig").\
        write.option("header", True)\
        .csv("data/phase2/validation/fraud_future.csv")

    print("Ok.")
