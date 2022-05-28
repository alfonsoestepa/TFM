import os

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType


def connect_to_spark():
    print("Connect to spark ...")
    conf = SparkConf()
    # conf.setMaster("spark://localhost:7077")
    # conf.set("spark.driver.host", "192.168.1.39")
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


def load_data_parquet(spark, filename):
    print("Loading data ...")
    return spark. \
        read. \
        parquet(filename). \
        withColumn("isFraud", col("isFraud").cast(DoubleType())). \
        withColumn("oldbalanceOrg", col("oldbalanceOrg").cast(DoubleType())). \
        withColumn("newbalanceOrig", col("newbalanceOrig").cast(DoubleType())). \
        withColumn("amount", col("amount").cast(DoubleType())). \
        withColumn("oldbalanceDest", col("oldbalanceDest").cast(DoubleType())). \
        withColumn("newbalanceDest", col("newbalanceDest").cast(DoubleType())). \
        withColumnRenamed("isFraud", "label")


def load_data_csv(spark, filename):
    print("Loading data ...")
    return spark. \
        read. \
        option("header", True). \
        csv(filename). \
        withColumn("isFraud", col("isFraud").cast(DoubleType())). \
        withColumn("oldBalanceOrig", col("oldBalanceOrig").cast(DoubleType())). \
        withColumn("newBalanceOrig", col("newBalanceOrig").cast(DoubleType())). \
        withColumn("amount", col("amount").cast(DoubleType())). \
        withColumn("oldBalanceDest", col("oldBalanceDest").cast(DoubleType())). \
        withColumn("newBalanceDest", col("newBalanceDest").cast(DoubleType())). \
        withColumnRenamed("isFraud", "label"). \
        withColumnRenamed("action", "type"). \
        withColumnRenamed("oldBalanceOrig", "oldbalanceOrg"). \
        withColumnRenamed("newBalanceOrig", "newbalanceOrig").drop("isUnauthorizedOverdraft")
