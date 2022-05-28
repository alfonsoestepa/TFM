from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


def evaluate_model(model, test_data, prefix=""):
    print("Evaluating the model ...")
    pred_pipeline = model.transform(test_data)

    evaluator = BinaryClassificationEvaluator()
    roc = evaluator.evaluate(pred_pipeline)

    evaluator = MulticlassClassificationEvaluator()
    return {
               prefix + "accuracy": evaluator.evaluate(pred_pipeline, {evaluator.metricName: "accuracy"}),
               prefix + "f1": evaluator.evaluate(pred_pipeline),
               prefix + "weightedPrecision": evaluator.evaluate(pred_pipeline,
                                                                {evaluator.metricName: "weightedPrecision"}),
               prefix + "weightedRecall": evaluator.evaluate(pred_pipeline, {evaluator.metricName: "weightedRecall"}),
               prefix + "AU-ROC": roc
           }, pred_pipeline


def calculate_confusion_matrix(predictions):
    return predictions.select(['prediction', 'label']).groupby("label").pivot(
        "prediction").count().drop("label").toPandas() / predictions.count()