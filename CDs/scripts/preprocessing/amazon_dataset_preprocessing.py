import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import random


class AmazonDataset():
    def __init__(self, preprocessing_args):
        super().__init__()
        self.args = preprocessing_args
        self.user_name_dict = {}  # rename users to integer names
        self.item_name_dict = {}

        self.users = []
        self.items = []

        # the interacted items for each user, sorted with date {user:[i1, i2, i3, ...], user:[i1, i2, i3, ...]}
        self.user_hist_inter_dict = {}
        # the rating matrix dict
        self.user_item_rating_dict = {}

        self.user_num = None
        self.item_num = None

        self.train_matrix = None
        self.train_matrix_origin = None
        self.test_data = None
        self.val_data = None
        self.pre_processing()
        self.split_dataset(seed=0)

    def get_user_item_dict(self, ):
        user_dict = {}
        item_dict = {}
        with open(self.args.ratings_dir, 'r') as f:
            line = f.readline().strip()
            while line:
                user = line.split('@')[0]
                item = line.split('@')[1]
                if user not in user_dict:
                    user_dict[user] = [item]
                else:
                    user_dict[user].append(item)
                if item not in item_dict:
                    item_dict[item] = [user]
                else:
                    item_dict[item].append(user)
                line = f.readline().strip()
        return user_dict, item_dict

    def pre_processing(self, ):
        user_dict, item_dict = self.get_user_item_dict()  # not sorted with time
        # {(user, item): rating, (user, item): rating ...}  # used to remove duplicate
        user_item_rating_dict = {}

        with open(self.args.ratings_dir, 'r', encoding='unicode_escape') as f:
            line = f.readline().strip()
            while line:
                user = line.split('@')[0]
                item = line.split('@')[1]
                rating = line.split('@')[2]
                if user in user_dict and item in user_dict[user] and (user, item) not in user_item_rating_dict:
                    user_item_rating_dict[(user, item)] = rating
                line = f.readline().strip()

        for key in list(user_item_rating_dict.keys()):
            if key[0] not in user_dict or key[1] not in user_dict[key[0]]:
                del user_item_rating_dict[key]

        # rename users, items, and topics to integer names
        user_name_dict = {}
        item_name_dict = {}

        count = 0
        for user in user_dict:
            if user not in user_name_dict:
                user_name_dict[user] = count
                count += 1
        count = 0
        for item in item_dict:
            if item not in item_name_dict:
                item_name_dict[item] = count
                count += 1

        renamed_user_item_rating_dict = {}
        for key, value in user_item_rating_dict.items():
            renamed_user_item_rating_dict[user_name_dict[key[0]],
                                          item_name_dict[key[1]]] = value
        user_item_rating_dict = renamed_user_item_rating_dict

        # {"u1": [i1, i2, i3, ...], "u2": [i1, i2, i3, ...]}, sort with time
        user_hist_inter_dict = {}

        for key, value in user_item_rating_dict.items():
            user = key[0]
            item = key[1]

            if user not in user_hist_inter_dict:
                user_hist_inter_dict[user] = [item]
            else:
                user_hist_inter_dict[user].append(item)

        users = list(set(user_name_dict.values()))
        items = list(set(item_name_dict.values()))

        self.user_name_dict = user_name_dict
        self.item_name_dict = item_name_dict
        self.user_hist_inter_dict = user_hist_inter_dict
        self.user_item_rating_dict = user_item_rating_dict
        self.users = users
        self.items = items
        self.user_num = len(users)
        self.item_num = len(items)
        return True

    def split_dataset(self, seed=0):
        random.seed(seed)
        row = []
        col = []
        data = []
        row_origin = []
        col_origin = []
        data_origin = []

        user_item_label_list = []  # [[u, [item1, item2, ...]]
        user_item_val_list = []
        for user, items in self.user_hist_inter_dict.items():
            training_pos_items, val_test_pos_items = train_test_split(
                items, test_size=0.3, random_state=seed)
            val_pos_items, test_pos_items = train_test_split(
                val_test_pos_items, test_size=2/3, random_state=seed)
            # test_pos_items = val_test_pos_items

            non_interacted_items = list(set(self.items) - set(items))
            add_moise_items = random.sample(
                non_interacted_items, len(test_pos_items))
            training_pos_items_new = list(
                set(training_pos_items) | set(add_moise_items))

            for item in training_pos_items_new:
                row = np.append(row, user)
                col = np.append(col, item)
                data = np.append(data, 1)

            for item in training_pos_items:
                row_origin = np.append(row_origin, user)
                col_origin = np.append(col_origin, item)
                data_origin = np.append(data_origin, 1)

            user_item_label_list.append(test_pos_items)
            user_item_val_list.append(val_pos_items)

        train_matrix_data = csr_matrix(
            (data, (row, col)), shape=(self.user_num,  self.item_num))
        train_matrix_origin_data = csr_matrix(
            (data_origin, (row_origin, col_origin)), shape=(self.user_num, self.item_num))
        self.train_matrix = train_matrix_data
        self.train_matrix_origin = train_matrix_origin_data
        print('# training samples :', self.train_matrix.count_nonzero())
        print('# test samples :', len(user_item_label_list))
        # self.test_data = np.array([np.array(t) for t in user_item_label_list])
        self.test_data = np.array(user_item_label_list, dtype=object)
        self.val_data = np.array(user_item_val_list, dtype=object)
        print("valid user: ", len(self.users))
        print('valid item : ', len(self.items))
        print('user dense is:', self.train_matrix.count_nonzero() / len(self.users))

        return True

    def save(self, save_path):
        return True

    def load(self):
        return False


def amazon_preprocessing(pre_processing_args):
    rec_dataset = AmazonDataset(pre_processing_args)
    return rec_dataset
