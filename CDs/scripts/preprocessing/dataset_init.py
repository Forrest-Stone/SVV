def dataset_init(preprocessing_args):
    if preprocessing_args.dataset == "cell_phones" or "kindle_store" or "electronic" or "cds_and_vinyl":
        from scripts.preprocessing.amazon_dataset_preprocessing import amazon_preprocessing
        rec_dataset = amazon_preprocessing(preprocessing_args)
    return rec_dataset
