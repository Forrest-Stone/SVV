import pandas as pd
from copy import deepcopy


# movielens datasets
# rating < 4 as negative data, just save the rating >= 4 scores
data = pd.read_csv('CDs_and_Vinyl.csv',
                   sep=',',
                   usecols=[0, 1, 2, 3],
                   names=["item_id", "user_id", "rating", "date"],
                   engine='python')

# data = data.drop(data[data["rating"] < 4].index)
print(data.head())
print(len(data))
data = data.drop_duplicates(subset=['user_id', 'item_id'])
print(len(data))


def remove_infrequent_items(data, min_counts=5):
    df = deepcopy(data)
    counts = df['item_id'].value_counts()
    df = df[df["item_id"].isin(counts[counts >= min_counts].index)]

    print("items with < {} interactoins are removed".format(min_counts))
    # print(df.describe())
    return df


def remove_infrequent_users(data, min_counts=10):
    df = deepcopy(data)
    counts = df['user_id'].value_counts()
    df = df[df["user_id"].isin(counts[counts >= min_counts].index)]

    print("users with < {} interactoins are removed".format(min_counts))
    # print(df.describe())
    return df


# because we need to sample the users interated items, so if we process users first, may be not
filtered_data = remove_infrequent_items(data, 20)
filtered_data = remove_infrequent_users(filtered_data, 12)

print('num of users:{}, num of items:{}'.format(
    len(filtered_data['user_id'].unique()),
    len(filtered_data['item_id'].unique())))
# filtered_data[["item_id", "user_id"]] = filtered_data[["user_id", "item_id"]]
print(filtered_data.head())

print(filtered_data["user_id"].value_counts())
print(filtered_data["item_id"].value_counts())

filtered_data = filtered_data.loc[:, ["user_id", "item_id", "rating", "date"]]
# filtered_data = filtered_data.sort_values("user_id", inplace=True)
print(filtered_data.head())

filtered_data.to_csv('CDs_ratings.txt', index=None, header=None, sep='@')
