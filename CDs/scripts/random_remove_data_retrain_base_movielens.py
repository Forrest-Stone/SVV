import torch
import numpy as np
import os
import tqdm
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from scripts.preprocessing.dataset_init import dataset_init
from utils.argument_amazon import arg_parse_train_base, arg_parser_preprocessing, arg_parse_div_optimize
from models.models import BaseRecModel
from models.data_loaders import AEWeigthDataset
from utils.evaluate_functions import evaluate_model, evaluate_model_origin_data
import logging
from time import time
from tensorboardX import SummaryWriter
from copy import deepcopy

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def use_weights(data, weights_data, test_data, th1, th2):
    weights_data_tmp = weights_data.clone()
    weights_data[:, :] = 1
    weights_data[data == 0] = 0
    for index, (row, tmp) in enumerate(zip(weights_data, weights_data_tmp)):
        # for row, tmp in zip(weights_data, weights_data_tmp):
        nonzero_indices = torch.nonzero(tmp).squeeze()
        # n = int(len(nonzero_indices) * th1)
        n = len(test_data[index])
        if len(nonzero_indices) >= n:
            random_indices = torch.randperm(len(nonzero_indices))[:n]
            indices_to_weights = nonzero_indices[random_indices]
            row[indices_to_weights] = 0
    data_tmp = weights_data

    return data_tmp

def statics(fastshap_values_para):
    div_args = arg_parse_div_optimize()

    # load data
    with open(os.path.join(div_args.data_obj_path, div_args.dataset + "_dataset_obj.pickle"), 'rb') as inp:
        rec_dataset = pickle.load(inp)

    train_matrix_origin = rec_dataset.train_matrix_origin.toarray()
    print("train_matrix_origin: ", train_matrix_origin)
    train_matrix_add_noise = rec_dataset.train_matrix.toarray()
    print("train_matrix_add_noise: ", train_matrix_add_noise)

    # load user 0 fashshap values
    # the values is a vector with the item form 0, to the max item id
    fastshap_values = fastshap_values_para


    overlaps = []
    for i in range(train_matrix_origin.shape[0]):
        # for i in range(5):
        nonzeros_index_of_origin = np.nonzero(train_matrix_origin[i])[0]
        nonzeros_index_of_add_noise = np.nonzero(train_matrix_add_noise[i])[0]
        noise_index = list(set(nonzeros_index_of_add_noise) -
                           set(nonzeros_index_of_origin))
        sort_nonzeros_index_of_fastshap = np.argsort(
            fastshap_values[i][nonzeros_index_of_add_noise])[:len(noise_index)]
        sort_nonzeros_index_of_fastshap = nonzeros_index_of_add_noise[
            sort_nonzeros_index_of_fastshap]
        # print("sort_nonzeros_index_of_fastshap: ",
        #       sort_nonzeros_index_of_fastshap)
        overlap_index = list(
            set(sort_nonzeros_index_of_fastshap) & set(noise_index))
        # print("overlap_index: ", overlap_index)
        overlap = len(overlap_index) / len(noise_index)
        overlaps.append(overlap)
    # print("overlaps: ", overlaps)
    print("mean overlap: ", np.mean(overlaps))

    overlaps = []
    for i in range(train_matrix_origin.shape[0]):
        # for i in range(5):
        nonzeros_index_of_origin = np.nonzero(train_matrix_origin[i])[0]
        nonzeros_index_of_add_noise = np.nonzero(train_matrix_add_noise[i])[0]
        noise_index = list(set(nonzeros_index_of_add_noise) -
                           set(nonzeros_index_of_origin))
        sort_nonzeros_index_of_fastshap = np.argsort(
            -fastshap_values[i][nonzeros_index_of_add_noise])[:len(noise_index)]
        sort_nonzeros_index_of_fastshap = nonzeros_index_of_add_noise[
            sort_nonzeros_index_of_fastshap]
        # print("sort_nonzeros_index_of_fastshap: ",
        #       sort_nonzeros_index_of_fastshap)
        overlap_index = list(
            set(sort_nonzeros_index_of_fastshap) & set(noise_index))
        # print("overlap_index: ", overlap_index)
        overlap = len(overlap_index) / len(noise_index)
        overlaps.append(overlap)
    # print("overlaps: ", overlaps)
    print("mean overlap: ", np.mean(overlaps))



def train_base_recommendation(train_args, pre_processing_args, div_args, th1, th2):
    if train_args.gpu:
        device = torch.device('cuda')
    else:
        device = 'cpu'

    if not os.path.exists(os.path.join(pre_processing_args.save_path,
                                       pre_processing_args.dataset + "_dataset_obj.pickle")):
        # get the input and output
        rec_dataset = dataset_init(pre_processing_args)
        Path(pre_processing_args.save_path).mkdir(parents=True, exist_ok=True)
        with open(
                os.path.join(pre_processing_args.save_path,
                             pre_processing_args.dataset + "_dataset_obj.pickle"),
                'wb') as outp:
            pickle.dump(rec_dataset, outp, pickle.HIGHEST_PROTOCOL)
    else:
        print("Load the pre-propcessing data.")
        with open(
                os.path.join(pre_processing_args.save_path,
                             pre_processing_args.dataset + "_dataset_obj.pickle"),
                'rb') as inp:
            rec_dataset = pickle.load(inp)

    data = torch.tensor(rec_dataset.train_matrix.toarray(),
                        dtype=torch.float32)
    # data = rec_dataset.train_matrix.toarray()
    # weights_data = np.load(os.path.join(
    #     div_args.save_path, div_args.div_type + '_fastshap_values_origin.npy'))
    # # weights_data = np.nan_to_num(weights_data)
    # weights_data = torch.tensor(weights_data, dtype=torch.float32)
    weights_data = data.clone()

    # the function to utilite the weights
    weights_data = use_weights(
        data, weights_data, rec_dataset.test_data, th1, th2)
    print(weights_data)
    print(train_args.weight_confidence)
    statics(weights_data)

    train_dataset = AEWeigthDataset(data, weights_data)
    train_loader = DataLoader(
        train_dataset, batch_size=train_args.batch_size, shuffle=True, num_workers=8)

    model = BaseRecModel(rec_dataset.user_num,
                         rec_dataset.item_num, train_args.drop_out).to(device)
    # model.load_state_dict(
    #     torch.load(
    #         os.path.join(div_args.base_model_path,
    #                      div_args.dataset + "_logs/base",
    #                      "best.base.model.pth")))

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=train_args.lr,
                                 weight_decay=train_args.weight_decay)
    loss_func = torch.nn.MSELoss(reduction='none')

    out_path = os.path.join("./logs", train_args.dataset + "_logs/base")
    Path(out_path).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(out_path + "/plots")

    t00 = time()
    init_metric = evaluate_model_origin_data(
        rec_dataset.test_data, weights_data, data, train_args.rec_k, model, device)
    t01 = time()
    output_eva_time = "[%.1f s]" % (t01 - t00)
    print("One evulation time: " + output_eva_time)
    logger.info(output_eva_time)
    print('init recall ndcg : ', init_metric)

    # For saving best model
    best_recall_ndcg = 0
    best_model = deepcopy(model)
    best_epoch = 0

    for epoch in tqdm.trange(1, train_args.epoch + 1):
        t0 = time()
        model.train()
        optimizer.zero_grad()
        train_losses = []
        for user_behaviour_feature, weights_x in train_loader:
            user_behaviour_feature = user_behaviour_feature.to(device)
            weights_x = weights_x.to(device)
            # # generate the mask matrix S
            S = torch.randint(2, user_behaviour_feature.shape).to(device)
            score = model.predict(weights_x, S).to(device)
            weight_confidence = 1 + weights_x * train_args.weight_confidence

            train_loss = (weight_confidence * loss_func(score,
                          weights_x)).sum().to(device)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss /= (len(user_behaviour_feature) * rec_dataset.item_num)
            train_losses.append(train_loss.to('cpu').detach().numpy())
            ave_train_loss = np.mean(np.array(train_losses))

        writer.add_scalar("Loss/train", ave_train_loss, epoch)

        if epoch % 5 == 0:
            model.eval()
            recall, ndcg = evaluate_model_origin_data(
                rec_dataset.test_data, weights_data, data, train_args.rec_k, model, device)
            output_str = "epoch %d: " % (epoch) + ", K = " + str(
                train_args.rec_k) + ", Recall: " + str(recall) + ", NDCG: " + str(ndcg)
            print(output_str)
            logger.info(output_str)

            writer.add_scalar("Evaluation/Recall10", recall[1], epoch)
            writer.add_scalar("Evaluation/NDCG10", ndcg[1], epoch)

            if best_recall_ndcg < (sum(recall) + sum(ndcg)):
                best_recall_ndcg = sum(recall) + sum(ndcg)
                best_epoch = epoch
                best_model = deepcopy(model)
                # torch.save(model.state_dict(), os.path.join(
                # out_path, "best.base.model.pth"))
                print("Best model saved at epoch: " + str(best_epoch))
                logger.info("Best model saved at epoch: " + str(best_epoch))
            else:
                if epoch - best_epoch > train_args.early_stop_epoch:
                    print("Early stopping at epoch: " + str(epoch))
                    logger.info("Early stopping at epoch: " + str(epoch))
                    break

        t1 = time()
        output_train = "Epoch %d: " % (
            epoch) + ", training loss: " + str(ave_train_loss) + ", time: %.1fs" % (t1 - t0)
        print(output_train)
        logger.info(output_train)

    model = best_model
    recall, ndcg = evaluate_model_origin_data(
        rec_dataset.test_data, weights_data, data, train_args.rec_k, model, device)
    output_str = "Best final epoch %d: " % (best_epoch) + ", K = " + str(
        train_args.rec_k) + ", Recall: " + str(recall) + ", NDCG: " + str(ndcg)
    print(output_str)
    logger.info(output_str)
    writer.close()
    logging.info("\n")
    logging.info("\n")
    torch.save(model.state_dict(), os.path.join(
        out_path, "retrain_best.base.model.pth"))
    return 0


if __name__ == "__main__":
    # set_seed(0)
    t_args = arg_parse_train_base()  # training arguments
    p_args = arg_parser_preprocessing()  # pre processing arguments
    d_args = arg_parse_div_optimize()  # diversity arguments
    if t_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = t_args.cuda
        print("Using CUDA", t_args.cuda)
    else:
        print("Using CPU")
    print(p_args)
    print(t_args)
    print(d_args)

    set_seed(0)
    train_base_recommendation(t_args, p_args, d_args, 0.0, 0.0)

    set_seed(1)
    train_base_recommendation(t_args, p_args, d_args, 0.0, 0.0)

    set_seed(2)
    train_base_recommendation(t_args, p_args, d_args, 0.0, 0.0)

    set_seed(3)
    train_base_recommendation(t_args, p_args, d_args, 0.0, 0.0)

    set_seed(2025)
    train_base_recommendation(t_args, p_args, d_args, 0.0, 0.0)
