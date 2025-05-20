import torch
import numpy as np
import os
import tqdm
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from scripts.preprocessing.dataset_init import dataset_init
from utils.argument_amazon import arg_parse_train_base, arg_parser_preprocessing
from models.models import BaseRecModel
from utils.evaluate_functions import evaluate_model
import logging
from time import time
from tensorboardX import SummaryWriter
from copy import deepcopy

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_base_recommendation(train_args, pre_processing_args):
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
    train_loader = DataLoader(
        data, batch_size=train_args.batch_size, shuffle=True, num_workers=8)

    model = BaseRecModel(rec_dataset.user_num,
                         rec_dataset.item_num, train_args.drop_out).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_args.lr, weight_decay=train_args.weight_decay)
    loss_func = torch.nn.MSELoss(reduction='none')

    out_path = os.path.join("./logs", train_args.dataset + "_logs/base")
    Path(out_path).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(out_path + "/plots")

    t00 = time()
    init_metric = evaluate_model(
        rec_dataset.test_data, rec_dataset.train_matrix, train_args.rec_k, model, device)
    t01 = time()
    output_eva_time = "[%.1f s]" % (t01 - t00)
    print("One evulation time: " + output_eva_time)
    logger.info(output_eva_time)
    print("init recall ndcg:", init_metric)

    # For saving best model
    best_recall_ndcg = 0
    best_model = deepcopy(model)
    best_epoch = 0

    for epoch in tqdm.trange(1, train_args.epoch + 1):
        t0 = time()
        model.train()
        optimizer.zero_grad()
        train_losses = []
        for user_behaviour_feature in train_loader:
            user_behaviour_feature = user_behaviour_feature.to(device)
            S = torch.randint(2, user_behaviour_feature.shape).to(device)
            score = model.predict(user_behaviour_feature, S).to(device)
            # score = model(user_behaviour_feature).to(device)
            # # COMPURT the weighted mse loss
            weight_confidence = 1 + user_behaviour_feature * train_args.weight_confidence
            train_loss = (weight_confidence * loss_func(score,
                          user_behaviour_feature)).sum().to(device)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss /= (len(user_behaviour_feature) * rec_dataset.item_num)
            train_losses.append(train_loss.to('cpu').detach().numpy())
            ave_train_loss = np.mean(np.array(train_losses))

        writer.add_scalar("Loss/train", ave_train_loss, epoch)

        if epoch % 5 == 0:
            model.eval()
            recall, ndcg = evaluate_model(
                rec_dataset.test_data, rec_dataset.train_matrix, train_args.rec_k, model, device)
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
    recall, ndcg = evaluate_model(
        rec_dataset.test_data, rec_dataset.train_matrix, train_args.rec_k, model, device)
    output_str = "Best final epoch %d: " % (best_epoch) + ", K = " + str(
        train_args.rec_k) + ", Recall: " + str(recall) + ", NDCG: " + str(ndcg)
    print(output_str)
    logger.info(output_str)
    writer.close()
    logging.info("\n")
    logging.info("\n")
    torch.save(model.state_dict(), os.path.join(
        out_path, "best.base.model.pth"))
    return 0


if __name__ == "__main__":
    # set_seed(0)
    t_args = arg_parse_train_base()  # training arguments
    p_args = arg_parser_preprocessing()  # pre processing arguments
    if t_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = t_args.cuda
        print("Using CUDA", t_args.cuda)
    else:
        print("Using CPU")
    print(p_args)
    print(t_args)

    set_seed(0)
    train_base_recommendation(t_args, p_args)
