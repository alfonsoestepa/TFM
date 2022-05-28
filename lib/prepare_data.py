def prepare_data(raw_dataset, seed, split_weights=[0.75, 0.25]):
    print("Prepare data ...")
    preprocessed_data = subsample_data(raw_dataset, seed)
    tr_data, test_data = preprocessed_data.randomSplit(split_weights, seed=seed)
    return preprocessed_data, tr_data, test_data


def subsample_data(raw_dataset, seed):
    fraud_op_count = raw_dataset[raw_dataset["label"] == 1].count()
    return raw_dataset.sampleBy("label", fractions={0: fraud_op_count / raw_dataset.count(), 1: 1}, seed=seed)
