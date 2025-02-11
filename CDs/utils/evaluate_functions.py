import numpy as np
import torch
import math


def evaluate_model(test_data, train_matrix, k, model, device):
    model.eval()
    num_batch_users = 1024
    # print(action_delta)
    recalls, ndcgs = [], []
    with torch.no_grad():
        num_users, num_items = train_matrix.shape
        number_batchs = int(num_users / num_batch_users) + 1
        user_indexes = np.arange(num_users)

        for batchID in range(number_batchs):
            start = batchID * num_batch_users
            end = min((batchID + 1) * num_batch_users, num_users)

            batch_user_index = user_indexes[start:end]
            # batch_user_index = torch.LongTensor(batch_user_index).to(device)
            # print(batch_user_index)
            user_input = torch.tensor(
                train_matrix.toarray()[batch_user_index], dtype=torch.float32).to(device)
            # S_input = torch.ones(
            #     user_input.shape, dtype=torch.float32).to(device)

            if model.__class__.__name__ == "BaseRecModel":
                # pred_scores = model(user_input, S_input)
                pred_scores = model(user_input)

            pred_scores = pred_scores.cpu().data.numpy().copy()

            # batch_user_index = batch_user_index.cpu().data.numpy()
            pred_scores[train_matrix[batch_user_index].toarray() > 0] = 0

            # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
            ind = np.argpartition(pred_scores, -k)
            ind = ind[:, -k:]
            arr_ind = pred_scores[np.arange(len(pred_scores))[:, None], ind]
            arr_ind_argsort = np.argsort(
                arr_ind)[np.arange(len(pred_scores)), ::-1]
            pred_items = ind[np.arange(len(pred_scores))[
                :, None], arr_ind_argsort]

            if batchID == 0:
                pred_list = pred_items.copy()
            else:
                pred_list = np.append(pred_list, pred_items, axis=0)

        for k in [5, 10, 20]:
            recalls.append(recall_at_k(test_data, pred_list, k))
            ndcgs.append(ndcg_k(test_data, pred_list, k))

    return recalls, ndcgs


def evaluate_model_origin_data(test_data, train_matrix, origin_data, k, model, device):
    model.eval()
    num_batch_users = 1024
    # print(action_delta)
    recalls, ndcgs = [], []
    with torch.no_grad():
        num_users, num_items = train_matrix.shape
        number_batchs = int(num_users / num_batch_users) + 1
        user_indexes = np.arange(num_users)

        for batchID in range(number_batchs):
            start = batchID * num_batch_users
            end = min((batchID + 1) * num_batch_users, num_users)

            batch_user_index = user_indexes[start:end]
            # batch_user_index = torch.LongTensor(batch_user_index).to(device)
            # print(batch_user_index)
            user_input = torch.tensor(
                train_matrix[batch_user_index], dtype=torch.float32).to(device)
            # S_input = torch.ones(
            #     user_input.shape, dtype=torch.float32).to(device)

            if model.__class__.__name__ == "BaseRecModel":
                # pred_scores = model(user_input, S_input)
                pred_scores = model(user_input)

            pred_scores = pred_scores.cpu().data.numpy().copy()

            # batch_user_index = batch_user_index.cpu().data.numpy()
            pred_scores[origin_data[batch_user_index] > 0] = 0

            # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
            ind = np.argpartition(pred_scores, -k)
            ind = ind[:, -k:]
            arr_ind = pred_scores[np.arange(len(pred_scores))[:, None], ind]
            arr_ind_argsort = np.argsort(
                arr_ind)[np.arange(len(pred_scores)), ::-1]
            pred_items = ind[np.arange(len(pred_scores))[
                :, None], arr_ind_argsort]

            if batchID == 0:
                pred_list = pred_items.copy()
            else:
                pred_list = np.append(pred_list, pred_items, axis=0)

        for k in [5, 10, 20]:
            recalls.append(recall_at_k(test_data, pred_list, k))
            ndcgs.append(ndcg_k(test_data, pred_list, k))

    return recalls, ndcgs


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def recall_at_k_array(actual, predicted, topk):
    sum_recall = []
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall.append(len(act_set & pred_set) / float(len(act_set)))
    return sum_recall


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in set(
            actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


def ndcg_k_array(actual, predicted, topk):
    res = []
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in set(
            actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res.append(dcg_k / idcg)
    return res


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res
