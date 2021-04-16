import math
import random
import pandas as pd
import numpy as np
from tqdm import tqdm


def loadData():
    path = '../../dataset/ml-latest-small/ratings.csv'
    data = pd.read_csv(path, usecols=[0, 1])
    user_dict = dict()
    item_dict=dict()

    for i in data.values:
        if i[0] in user_dict:
            user_dict[i[0]].add(i[1])
        else:
            user_dict[i[0]] = {i[1]}

    for i in data.values:
        if i[1] in item_dict:
            item_dict[i[1]].add(i[0])
        else:
            item_dict[i[1]] = {i[0]}
    return user_dict,item_dict

def trainsetSpilt(dct):

    train_set, test_set = dict(), dict()
    for uid in dct:
        test_set[uid] = set(random.sample(dct[uid], math.ceil(0.2 * len(dct[uid]))))
        train_set[uid] = dct[uid] - test_set[uid]
    return train_set, test_set


def cossim(s1, s2):
    #compute cos value
    return len(s1 & s2) / (len(s1) * len(s2)) ** 0.5


def knn(train_set, k):
    user_sims = {}
    for u1 in tqdm(train_set):
        user_list = []
        for u2 in train_set:
            if u1 == u2 or len(train_set[u1] & train_set[u2]) == 0:
                continue
            rate = cossim(train_set[u1], train_set[2])
            user_list.append({'id': u2, 'rate': rate})

        user_sims[u1] = sorted(user_list, key=lambda user_list: user_list['rate'], reverse=True)[:k]
    return user_sims

def recommendMoviesByUserCF(user_sims,data):
    recommendation=dict()
    for u in tqdm(user_sims):
        recommendation[u]=set()
        for sim in user_sims[u]:
            #distinct value
            recommendation[u] |=(data[sim['id']]-data[u])
    return recommendation

def recommendMoviesByItemCF(item_sims,data):
    recommendation=dict()
    for u in tqdm(data):
        recommendation[u]=set()
        for item in data[u]:
            recommendation[u] |= set(i['id'] for i in item_sims[item]) - data[u]
    return recommendation

def precision_recall(pre,test):
    precision=0
    recall=0
    for uid in test:
        t=len(pre[uid]&test[uid])
        precision+=t/(len(pre[uid])+1)
        recall+=t/(len(test[uid])+1)
    return precision/len(test),recall/len(test)

def main():
    user_dict,item_dict= loadData()
    train_set, test_set = trainsetSpilt(user_dict)
    train_set_item,test_set_item=trainsetSpilt(item_dict)
    user_sims = knn(train_set, 5)
    item_sims=knn(train_set_item,5)
    pre_set=recommendMoviesByUserCF(user_sims,train_set)
    pre_item_set=recommendMoviesByItemCF(item_sims,train_set_item)
    p,r=precision_recall(pre_set,test_set)
    p_item,r_item=precision_recall(pre_item_set,test_set_item)
    print(p,r)
    print(p_item,r_item)


if __name__ == '__main__':
    main()
