import torch
from torch import nn
import torch.nn.functional as F


class BaseRecModel(torch.nn.Module):
    def __init__(self, user_num, item_num, drop_out):
        super(BaseRecModel, self).__init__()
        self.user_number = user_num
        self.item_number = item_num
        self.drop_out = drop_out
        # ecoder
        self.fc1 = nn.Linear(self.item_number, 50)
        # decoder
        self.fc4 = nn.Linear(50, self.item_number)

        # dropout with 0.1 drop probability
        self.dropout = nn.Dropout(self.drop_out)

    def forward(self, user_feature):
        x = F.relu(self.fc1(user_feature))
        x = torch.sigmoid(self.fc4(x))
        return x

    def predict(self, user_input, S):
        user_input = user_input * S + 0 * (1 - S)
        x = F.relu(self.fc1(user_input))
        x = torch.sigmoid(self.fc4(x))
        return x
