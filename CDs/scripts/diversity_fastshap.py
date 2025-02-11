from fastshap import FastSHAP
import torch
import numpy as np
import pickle
import os
from pathlib import Path
from utils.argument_amazon import arg_parse_div_optimize
from models.models import BaseRecModel
from utils.evaluate_functions import evaluate_model, evaluate_val_set_batch_user
from time import time
import datetime
from sklearn.model_selection import train_test_split
import torch.nn as nn
import logging
import subprocess
import atexit

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    set_seed(0)
    div_args = arg_parse_div_optimize()
    if div_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = div_args.cuda
        print("Using CUDA", div_args.cuda)
    else:
        print("Using CPU")
    print(div_args)

    if div_args.gpu:
        device = torch.device('cuda')
    else:
        device = 'cpu'

    # import dataset
    with open(
            os.path.join(div_args.data_obj_path,
                         div_args.dataset + "_dataset_obj.pickle"),
            'rb') as inp:
        rec_dataset = pickle.load(inp)

    # generate the output y from base model
    base_model = BaseRecModel(rec_dataset.user_num,
                              rec_dataset.item_num, div_args.drop_out).to(device)
    base_model.load_state_dict(
        torch.load(
            os.path.join(div_args.base_model_path,
                         div_args.dataset + "_logs/base",
                         "best.base.model.pth")))
    base_model.eval()

    #  fix the rec model
    for name, param in base_model.named_parameters():
        param.requires_grad = False

    t00 = time()
    recall, ndcg = evaluate_model(
        rec_dataset.test_data, rec_dataset.train_matrix, div_args.rec_k, base_model, device)
    t01 = time()
    output_eva_time = "[%.1f s]" % (t01 - t00)
    print("One evulation time: " + output_eva_time)
    logger.info(output_eva_time)
    output_str = "Recall: " + str(recall) + ", NDCG: " + str(ndcg)
    print(output_str)
    logging.info(output_str)

    Path(div_args.save_path).mkdir(parents=True, exist_ok=True)

    # prepare the explainer model data
    data = torch.tensor(rec_dataset.train_matrix.toarray(),
                        dtype=torch.float32)
    # data_val_label = torch.zeros(
    #     data.shape[0], data.shape[1], dtype=torch.float32)
    # for i in range(rec_dataset.val_data.shape[0]):
    #     data_val_label[i][rec_dataset.val_data[i]] = 1
    # data = torch.cat((data, data_val_label), dim=1)
    train, val_test = train_test_split(data, test_size=0.2, random_state=0)

    # Setup
    num_features = rec_dataset.item_num

    # Set up imputer object
    class Imputer:
        def __init__(self):
            self.num_players = num_features

        def __call__(self, x, S):
            # Call surrogate model (with data normalization)
            # x = x[:, :self.num_players].to(device)
            # base_predict_x = base_model(x)
            predict_x = base_model.predict(x, S)
            value_f = (predict_x * x).sum(dim=1)
            y = value_f / torch.count_nonzero(x, dim=1)
            y = y.unsqueeze(-1)
            return y

    imputer = Imputer()

    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(start_time)
    logging.info(start_time)

    # Calculate FastSHAP values
    Path(div_args.save_path).mkdir(parents=True, exist_ok=True)
    if os.path.isfile(os.path.join(div_args.save_path, div_args.div_type + '_explainer.pth')):
        print("Loading saved explainer model.")
        explainer = torch.load(os.path.join(
            div_args.save_path, div_args.div_type + '_explainer.pth')).to(device)
        fastshap = FastSHAP(explainer, imputer,
                            normalization='additive', link='none')
        # fastshap = FastSHAP(explainer, imputer,
        #                     normalization='multiplicative', link='none')

    else:
        # Create explainer model
        explainer = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_features)).to(device)

        # Set up FastSHAP object
        fastshap = FastSHAP(explainer, imputer,
                            normalization='additive', link='none')

        # train
        fastshap.train(train, val_test, lr=1e-1, eff_lambda=0, batch_size=256,
                       num_samples=32, max_epochs=300, validation_samples=32, verbose=True)

        # Save explainer
        explainer.cpu()
        torch.save(explainer, os.path.join(div_args.save_path,
                   div_args.div_type + '_explainer.pth'))
        explainer.to(device)

    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(end_time)
    logging.info(end_time)

    # Calculate FastSHAP values
    fastshap_values = fastshap.shap_values(data[0:1])[0]
    print(fastshap_values)

    fastshap_values = fastshap.shap_values(data)
    print("fastshap_values: ")
    print(fastshap_values.squeeze())
    np.save(os.path.join(div_args.save_path, div_args.div_type +
            '_fastshap_values_origin.npy'), fastshap_values.squeeze() * rec_dataset.train_matrix.toarray())

    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(end_time)
    logging.info(end_time)
