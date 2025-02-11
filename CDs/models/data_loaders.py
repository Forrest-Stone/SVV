from torch.utils.data import Dataset


class UserItemInterDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        user = self.data[index][0]
        pos_item = self.data[index][1]
        neg_item = self.data[index][2]
        return user, pos_item, neg_item

    def __len__(self):
        return len(self.data)


class AEWeigthDataset(Dataset):
    def __init__(self, data, weight_confidence):
        self.data = data
        self.weight_confidence = weight_confidence

    def __getitem__(self, index):
        input_x = self.data[index]
        weight_x = self.weight_confidence[index]
        return input_x, weight_x

    def __len__(self):
        return len(self.data)
