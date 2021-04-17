import os
import pandas as pd
import numpy as np
from pprint import pprint

DATA_PATH = '../../dataset/ml-latest-small/ratings.csv'
CACHE_DIR = '../../dataset/cache/'


def load_data(data_path):
    cache_path = os.path.join(CACHE_DIR, 'ratings_matrix.cache')
    if os.path.exists(cache_path):
        ratings_matrix = pd.read_pickle(cache_path)
    else:
        dtype = {'userId': np.int32, 'movieId': np.int32, 'rating': np.float32}
        ratings = pd.read_csv(data_path, dtype=dtype, usecols=range(3))
        # values要不能用括号不然会出bug
        ratings_matrix = ratings.pivot_table(index=["userId"], columns=["movieId"], values="rating")
        ratings_matrix.to_pickle(cache_path)
    return ratings_matrix


def pearson_similarity(ratings_matrix, based='user'):
    user_sim_cache_path = os.path.join(CACHE_DIR, 'user_sim.cache')
    item_sim_cache_path = os.path.join(CACHE_DIR, 'item_sim.cache')

    if based == 'user':
        if os.path.exists(user_sim_cache_path):
            sim = pd.read_pickle(user_sim_cache_path)
        else:
            sim = ratings_matrix.T.corr()
            sim.to_pickle(user_sim_cache_path)
    elif based == 'item':
        if os.path.exists(item_sim_cache_path):
            sim = pd.read_pickle(item_sim_cache_path)
        else:
            sim = ratings_matrix.corr()
            sim.to_pickle(item_sim_cache_path)
    else:
        raise Exception("Please input user or item")
    return sim


def predict_user_based(uid, iid, ratings_matrix, user_sim):
    sim_user = user_sim.loc[uid].drop([uid]).dropna()
    sim_user = sim_user.where(sim_user > 0).dropna()
    if sim_user.empty is True:
        raise Exception('no similarity user')
    ids = set(ratings_matrix.loc[:, iid].dropna().index) & set(sim_user.index)
    final_user = sim_user.loc[list(ids)]

    sum_up = 0.
    sum_down = 0.
    for sim_uid, sim_score in final_user.items():
        sum_down += sim_score
        sum_up += ratings_matrix.loc[sim_uid, iid] * sim_score
    predict_rating = sum_up / sum_down
    return round(predict_rating, 2)


def _predict_all_user_based(uid, item_ids, ratings_matrix, user_sim):
    for iid in item_ids:
        try:
            rating = predict_user_based(uid, iid, ratings_matrix, user_sim)
        except Exception as e:
            pass
        else:
            yield uid, iid, rating


def predict_all_user_based(uid, ratings_matrix, user_sim, filter_rule=None):
    # 基于某项规则进行召回
    # 'unhot':过滤不热门的电影
    # 'rated':过滤打过分的电影

    if not filter_rule:
        item_ids = ratings_matrix.columns
    elif filter_rule == 'unhot':
        count = ratings_matrix.count()
        item_ids = count.where(count > 10).dropna().index
    elif filter_rule == 'rated':
        user_rating = ratings_matrix.loc[uid]
        ids = user_rating[user_rating < 6].dropna().index
        item_ids = user_rating.drop(ids).index
    elif set(filter_rule) == set(['unhot', 'rated']):
        count = ratings_matrix.count()
        ids1 = count.where(count > 10).dropna().index
        user_rating = ratings_matrix.loc[uid]
        _ids = user_rating[user_rating < 6].dropna().index
        ids2 = user_rating.drop(_ids).index
        item_ids = set(ids1) & set(ids2)
    else:
        raise Exception('no such rule')

    yield from _predict_all_user_based(uid, item_ids, ratings_matrix, user_sim)


def predict_item_based(uid, iid, ratings_matrix, item_sim):
    sim_item = item_sim.loc[iid].drop([iid]).dropna()
    sim_item = sim_item.where(sim_item > 0).dropna()
    if sim_item.empty is True:
        raise Exception("no similarity item")
    ids = set(ratings_matrix.loc[uid].dropna().index) & set(sim_item.index)
    final_ids = sim_item.loc[list(ids)]
    sum_up = 0.
    sum_down = 0.
    for iid, sim_score in final_ids.items():
        sum_down += sim_score
        sum_up += ratings_matrix.loc[uid, iid] * sim_score
    predict_rating = sum_up / sum_down
    return round(predict_rating, 2)


def _predict_all_item_based(uid, item_ids, ratings_matrix, item_sim):
    for iid in item_ids:
        try:
            rating = predict_item_based(uid, iid, ratings_matrix, item_sim)
        except Exception as e:
            pass
        else:
            yield uid, iid, rating


def predict_all_item_based(uid, ratings_matrix, item_sim, filter_rule=None):
    if not filter_rule:
        item_ids = ratings_matrix.columns
    elif filter_rule == 'unhot':
        count = ratings_matrix.count()
        item_ids = count.where(count > 10).dropna().index
    elif filter_rule == 'rated':
        user_rating = ratings_matrix.loc[uid]
        ids = user_rating.where(user_rating < 6).dropna().index
        item_ids = user_rating.drop(ids).index
    elif set(filter_rule) == set(['unhot', 'rated']):
        count = ratings_matrix.count()
        ids1 = count.where(count > 10).dropna().index
        user_rating = ratings_matrix.loc[uid]
        _ids = user_rating.where(user_rating < 6).dropna().index
        ids2 = user_rating.drop(_ids).index
        item_ids = set(ids1) & set(ids2)
    else:
        raise Exception('no such rule')

    yield from _predict_all_item_based(uid, item_ids, ratings_matrix, item_sim)


def top_N_rs(uid, n, based='user', filter_rule=None):
    # 基于预测评分进行排序
    ratings_matrix = load_data(DATA_PATH)
    if based == 'user':
        user_sim = pearson_similarity(ratings_matrix, 'user')
        results = predict_all_user_based(uid, ratings_matrix, user_sim, filter_rule)
    elif based == 'item':
        item_sim = pearson_similarity(ratings_matrix, 'item')
        results = predict_all_item_based(uid, ratings_matrix, item_sim, filter_rule)
    return sorted(results, key=lambda x: x[2], reverse=True)[:n]


if __name__ == '__main__':
    n = 5
    uid = 1
    filter_rule = ['unhot', 'rated']
    results = np.array(top_N_rs(uid, n, 'user', filter_rule), dtype=np.int32)[:, 1]
    print('基于User-Based CF推荐给 用户%d 的%d部电影ID:' % (uid, n))
    print(results)
    results = np.array(top_N_rs(uid, n, 'item', filter_rule), dtype=np.int32)[:, 1]
    print('基于Item-Based CF推荐给 用户%d 的%d部电影ID:' % (uid, n))
    print(results)
