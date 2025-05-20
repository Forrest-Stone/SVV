import argparse


def arg_parser_preprocessing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        dest="dataset",
                        type=str,
                        default="cds_and_vinyl")
    parser.add_argument("--ratings_dir",
                        dest="ratings_dir",
                        type=str,
                        default="./datasets/CDs/CDs_ratings.txt",
                        help="path to pre-extracted ratings data")
    parser.add_argument("--split_ratio",
                        dest="split_ratio",
                        type=int,
                        default=0.2,
                        help="split the datasets")
    parser.add_argument(
        "--save_path",
        dest="save_path",
        type=str,
        default="./dataset_objs/",
        help="The path to save the preprocessed dataset object")
    return parser.parse_args()


def arg_parse_train_base():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        dest="dataset",
                        type=str,
                        default="cds_and_vinyl")
    parser.add_argument("--gpu",
                        dest="gpu",
                        action="store_false",
                        help="whether to use gpu")
    parser.add_argument("--cuda",
                        dest="cuda",
                        type=str,
                        default='4',
                        help="which cuda")
    parser.add_argument("--weight_decay",
                        dest="weight_decay",
                        type=float,
                        default=1e-3,
                        help="L2 norm to the wights")
    parser.add_argument("--lr",
                        dest="lr",
                        type=float,
                        default=1e-3,
                        help="learning rate for training")
    parser.add_argument("--epoch",
                        dest="epoch",
                        type=int,
                        default=200,
                        help="training epoch")
    parser.add_argument("--batch_size",
                        dest="batch_size",
                        type=int,
                        default=256,
                        help="batch size for training base rec model")
    parser.add_argument("--drop_out", dest="drop_out", type=float,
                        default=0.1, help="drop probability rate")
    parser.add_argument(
        "--weight_confidence",
        dest="weight_confidence",
        type=int,
        default=24,
        help="the weight of the rating entry")
    parser.add_argument("--early_stop_epoch",
                        dest="early_stop_epoch",
                        type=int,
                        default=10,
                        help="early_stop_epoch")
    parser.add_argument("--rec_k",
                        dest="rec_k",
                        type=int,
                        default=20,
                        help="length of rec list")
    return parser.parse_args()


def arg_parse_div_optimize():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        dest="dataset",
                        type=str,
                        default="cds_and_vinyl")
    parser.add_argument("--base_model_path",
                        dest="base_model_path",
                        type=str,
                        default="./logs/")
    parser.add_argument("--gpu",
                        dest="gpu",
                        action="store_false",
                        help="whether to use gpu")
    parser.add_argument("--cuda",
                        dest="cuda",
                        type=str,
                        default='4',
                        help="which cuda")
    parser.add_argument(
        "--data_obj_path",
        dest="data_obj_path",
        type=str,
        default="./dataset_objs/",
        help="the path to the saved dataset object in the training phase")
    parser.add_argument("--value_type",
                        dest="value_type",
                        type=str,
                        default="NDCG",
                        help="which value function type we want to obserave, Recall or NDCG")
    parser.add_argument("--rec_k",
                        dest="rec_k",
                        type=int,
                        default=20,
                        help="length of rec list to evulation")
    parser.add_argument("--drop_out", dest="drop_out", type=float,
                        default=0.1, help="drop probability rate")
    parser.add_argument("--save_path",
                        dest="save_path",
                        type=str,
                        default="./fastshap_results/",
                        help="save the results of fastshap model")
    return parser.parse_args()
