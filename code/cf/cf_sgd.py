import numpy as np
import pandas as pd

DATA_PATH = '../../dataset/ml-latest-small/ratings.csv'


class BaselineCfBySGD(object):
    def __init__(self, epochs, learning_rate, reg, columns=['uid', 'iid', 'rating']):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.reg = reg
        self.columns = columns

    def fit(self, dataset):
        self.dataset = dataset
        self.user_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.item_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]
        self.global_mean = self.dataset[self.columns[2]].mean()
        self.bu, self.bi = self.sgd()

    def sgd(self):
        bu = dict(zip(self.user_ratings.index, np.zeros(len(self.user_ratings))))
        bi = dict(zip(self.item_ratings.index, np.zeros(len(self.item_ratings))))

        for i in range(self.epochs):
            for uid, iid, real_rating in self.dataset.itertuples(index=False):
                error = real_rating - (self.global_mean + bu[uid] + bi[iid])
                bu[uid] += self.learning_rate * (error - self.reg * bu[uid])
                bi[iid] += self.learning_rate * (error - self.reg * bi[iid])

        return bu, bi

    def predict(self, uid, iid):
        predict_rating = self.global_mean + self.bu[uid] + self.bi[iid]
        return predict_rating

    def test(self, testset):
        for uid, iid, real_rating in testset.itertuples(index=False):
            try:
                pred_rating = self.predict(uid, iid)
            except Exception as e:
                pass
            else:
                yield uid, iid, real_rating, pred_rating


def data_split(data_path, x=0.8):
    dtype = {'userId': np.int32, 'movieId': np.int32, 'rating': np.float32}
    dataset = pd.read_csv(data_path, dtype=dtype, usecols=range(3))
    testset_index = []
    for uid in dataset.groupby('userId').any().index:
        user_rating_data = dataset.where(dataset['userId'] == uid).dropna()
        index = round(len(user_rating_data) * x)
        testset_index += list(user_rating_data.index.values[index:])
    testset = dataset.loc[testset_index]
    trainset = dataset.drop(testset_index)

    return trainset, testset


def accuracy(predict_results, method='all'):
    def rmse(predict_results):
        length = 0
        _sum = 0
        for uid, iid, real_rating, predict_results in predict_results:
            length += 1
            _sum += (predict_results - real_rating) ** 2
        return round(np.sqrt(_sum / length), 4)

    def mae(predict_results):
        length = 0
        _sum = 0
        for uid, iid, real_rating, predict_results in predict_results:
            length += 1
            _sum += np.abs(predict_results - real_rating)
        return round(_sum / length, 4)

    def rmse_mae(predict_results):
        length = 0
        _rmse_sum = 0
        _mae_sum = 0
        for uid, iid, real_rating, predict_results in predict_results:
            length += 1
            _rmse_sum += (predict_results - real_rating) ** 2
            _mae_sum += np.abs(predict_results - real_rating)
        return round(np.sqrt(_rmse_sum / length), 4), round(_mae_sum / length, 4)

    if method.lower() == 'rmse':
        rmse(predict_results)
    elif method.lower() == 'mae':
        mae(predict_results)
    else:
        return rmse_mae(predict_results)


if __name__ == '__main__':
    trainset, testset = data_split(DATA_PATH)
    bcf = BaselineCfBySGD(500, 0.1, 0.1, ['userId', 'movieId', 'rating'])
    bcf.fit(trainset)
    pred_results = bcf.test(testset)
    rmse, mae = accuracy(pred_results)
    print("rmse:", rmse, "  mas:", mae)
