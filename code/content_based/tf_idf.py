import numpy as np
import pandas as pd
import collections
from pprint import pprint
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from functools import reduce



def get_movie_dataset():
    _tag = pd.read_csv('../../dataset/ml-latest-small/tags.csv', usecols=range(1, 3)).dropna()
    tags = _tag.groupby('movieId').agg(list)

    movies = pd.read_csv('../../dataset/ml-latest-small/movies.csv', index_col='movieId')
    movies["genres"] = movies["genres"].apply(lambda x: x.split("|"))

    movies_index = set(movies.index) & set(tags.index)
    new_tags = tags.loc[list(movies_index)]
    ret = movies.join(new_tags)

    movies_dataset = pd.DataFrame(
        map(lambda x: (x[0], x[1], x[2], x[2] + x[3]) if x[3] is not np.nan else (x[0], x[1], x[2], []),
            ret.itertuples()),
        columns=['movieId', 'title', 'genres', 'tags'])
    movies_dataset.set_index("movieId", inplace=True)
    return movies_dataset


def creat_movie_profile(movie_dataset):
    dataset = movie_dataset['tags'].values

    dct = Dictionary(dataset)
    corpus = [dct.doc2bow(line) for line in dataset]
    model = TfidfModel(corpus)

    _movie_profile = []
    for i, data in enumerate(movie_dataset.itertuples()):
        mid = data[0]
        title = data[1]
        genres = data[2]
        vector = model[corpus[i]]
        movie_tags = sorted(vector, key=lambda x: x[1], reverse=True)[:30]
        topN_tags_weights = dict(map(lambda x: (dct[x[0]], x[1]), movie_tags))
        for g in genres:
            topN_tags_weights[g] = 1.0
        topN_tags = [i[0] for i in topN_tags_weights.items()]
        _movie_profile.append((mid, title, topN_tags, topN_tags_weights))
    movie_profile = pd.DataFrame(_movie_profile, columns=['movieId', 'title', 'profile', 'weights'])
    movie_profile.set_index('movieId', inplace=True)
    return movie_profile


def create_inverted_table(movie_profile):
    inverted_table = {}
    for mid, weights in movie_profile['weights'].iteritems():
        for tag, weight in weights.items():
            _ = inverted_table.get(tag, [])
            _.append((mid, weight))
            inverted_table.setdefault(tag, _)
    return inverted_table


def create_user_profile(movie_profile):
    record = pd.read_csv('../../dataset/ml-latest-small/ratings.csv', usecols=range(2),
                         dtype={'userId': np.int32, 'movieId': np.int32})

    record = record.groupby('userId').agg(list)

    user_profile = {}
    for uid, mids in record.itertuples():
        record_movie_profile = movie_profile.loc[list(mids)]
        counter = collections.Counter(reduce(lambda x, y: list(x) + list(y), record_movie_profile['profile'].values))
        interest_word = counter.most_common(50)
        maxcount = interest_word[0][1]
        interest_word = [(w, round(c / maxcount, 4)) for w, c in interest_word]
        user_profile[uid] = interest_word
    return user_profile

def recommend(uid, k):
    movie_dataset = get_movie_dataset()
    movie_profile = creat_movie_profile(movie_dataset)
    inverted_table = create_inverted_table(movie_profile)
    user_profile = create_user_profile(movie_profile)
    interest_words = user_profile[uid]
    result_table = {}
    for interest_word, interest_weight in interest_words:
        related_movies = inverted_table[interest_word]
        for mid, related_weight in related_movies:
            _ = result_table.get(mid, [])
            #_.append(interest_weight) #用户兴趣程度
            _.append(related_weight)#兴趣词与电影的关联程度
            #_.append(interest_weight*related_weight)
            result_table.setdefault(mid, _)
    rs_result = map(lambda x: (x[0], sum(x[1])), result_table.items())
    rs_result = sorted(rs_result, key=lambda x: x[1], reverse=True)[:k]
    return rs_result

if __name__ == '__main__':
    res = recommend(1, 10)
    pprint(res)
