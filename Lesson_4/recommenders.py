import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекомендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, weighting=True):
        
        # your_code. Это необязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        
        self.data = data
        self.user_item_matrix = self.prepare_matrix(self.data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
     
    @staticmethod
    def prepare_matrix(data: pd.DataFrame):
        
        # your_code
        
        user_item_matrix = pd.pivot_table(data=data.loc[data['quantity'] != 0],
                                          index='user_id',
                                          columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0,
                                          sort=False)
                            
        
        return user_item_matrix.astype(float)
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари
           Returns id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
        """
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr(),
                            show_progress=False)
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=0, random_state=0):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors, 
                                        regularization=regularization,
                                        iterations=iterations,  
                                        num_threads=num_threads,
                                        random_state=random_state)
        model.fit(csr_matrix(user_item_matrix).T.tocsr(),
                  show_progress=False)
        
        return model

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        # your_code
        
        popularity = self.data.groupby(['user_id', 'item_id'], sort=False)['quantity'].count().reset_index()
        popularity.sort_values('quantity', ascending=False, inplace=True)
        popularity = popularity.groupby('user_id').head(N)
        
        items = popularity.loc[popularity['user_id'] == user, 'item_id'].tolist()
        recs = [self.id_to_itemid[self.model.similar_items(self.itemid_to_id[item], N=2)[1][0]] for item in items]
        
        # Есть пользователи, которые могли купить меньше, чем N разных товаров.
        # Так как в конце функции нам установлена проверка на количество рекомендаций,
        # то будем заполнять недостающие рекомендации похожими товарам на топ-1 для
        # таких юзеров
        if len(recs) < N:
            diff = N - len(recs)
            recs += [self.id_to_itemid[self.model.similar_items(self.itemid_to_id[items[0]], N=diff+2)[i][0]] for i in range(2, diff+2)]

        assert len(recs) == N, 'Количество рекомендаций != {}'.format(N)
        return recs
    
    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        # your_code
        
        popularity = self.data.groupby(['user_id', 'item_id'], sort=False)['quantity'].count().reset_index()
        popularity.sort_values('quantity', ascending=False, inplace=True)
        popularity = popularity.groupby('user_id').head(1)
        
        users = [self.id_to_userid[self.model.similar_users(self.userid_to_id[user], N=N+1)[i][0]] for i in range(1, N+1)]
        recs = popularity.loc[popularity['user_id'].isin(users), 'item_id'].tolist()
        
        assert len(recs) == N, 'Количество рекомендаций != {}'.format(N)
        return recs