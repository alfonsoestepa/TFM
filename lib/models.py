from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.pipeline import Pipeline


def train_logistic_regression(train_data, **kwargs):
    print("Train model ...")
    name_encoder = StringIndexer(inputCols=["nameOrig", "nameDest", "type"],
                                 outputCols=["nameOrigIdx", "nameDestIdx", "typeIdx"], handleInvalid="keep")
    type_encoder = OneHotEncoder(inputCol="typeIdx", outputCol="typeEnc")
    assembler_tr = VectorAssembler(
        inputCols=["amount", "oldbalanceOrg", "oldBalanceDest", "newbalanceOrig", "newBalanceDest", "typeEnc", "nameOrigIdx", "nameDestIdx"],
        # inputCols=["amount", "oldbalanceOrg", "newbalanceOrig", "typeEnc"],
        outputCol="features")
    model = LogisticRegression(**kwargs, featuresCol="features", labelCol="label")

    pipeline_es = Pipeline(stages=[name_encoder, type_encoder, assembler_tr, model])
    pipeline_model = pipeline_es.fit(train_data)
    return pipeline_model, "logistic-regression", pipeline_model.stages[-1].coefficients


def train_random_forrest(train_data, **kwargs):
    print("Train model ...")
    name_encoder = StringIndexer(inputCols=["nameOrig", "nameDest", "type"],
                                 outputCols=["nameOrigIdx", "nameDestIdx", "typeIdx"], handleInvalid="keep")
    type_encoder = OneHotEncoder(inputCol="typeIdx", outputCol="typeEnc")
    assembler_tr = VectorAssembler(
        # inputCols=["step", "amount", "oldbalanceOrg", "oldbalanceDest", "newbalanceOrig", "newbalanceDest", "typeEnc"],
        inputCols=["amount", "oldbalanceOrg", "newbalanceOrig", "typeEnc"],
        outputCol="features")
    model = RandomForestClassifier(**kwargs)

    pipeline_es = Pipeline(stages=[name_encoder, type_encoder, assembler_tr, model])
    pipeline_model = pipeline_es.fit(train_data)
    return pipeline_model, "random-forrest", pipeline_model.stages[-1].featureImportances


def train_linear_svc(train_data, **kwargs):
    print("Train model ...")
    name_encoder = StringIndexer(inputCols=["nameOrig", "nameDest", "type"],
                                 outputCols=["nameOrigIdx", "nameDestIdx", "typeIdx"], handleInvalid="keep")
    type_encoder = OneHotEncoder(inputCol="typeIdx", outputCol="typeEnc")
    assembler_tr = VectorAssembler(
        inputCols=["amount", "oldbalanceOrg", "oldBalanceDest", "newbalanceOrig", "newBalanceDest", "typeEnc", "nameOrigIdx", "nameDestIdx"],
        #inputCols=["amount", "oldbalanceOrg", "newbalanceOrig", "typeEnc"],
        outputCol="features")
    model = LinearSVC(**kwargs)

    pipeline_es = Pipeline(stages=[name_encoder, type_encoder, assembler_tr, model])
    pipeline_model = pipeline_es.fit(train_data)
    return pipeline_model, "linear-svc", pipeline_model.stages[-1].coefficients
