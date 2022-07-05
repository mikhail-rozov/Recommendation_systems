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
    
    def __init__(self, data, weighting=True, K1=100, B=0.8):
        
        self.data = data
        self.user_item_matrix = self.prepare_matrix(self.data)
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'], sort=False)['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]
        
        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id', sort=False)['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T, K1=K1, B=B).T 
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
     
    @staticmethod
    def prepare_matrix(data: pd.DataFrame):
        
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

    def _extend_with_top_popular(self, recs, N=5):
        if len(recs) < N:
            diff = N - len(recs)
            recs += self.overall_top_purchases[:diff]
    
        return recs
    
    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        
        top_user_purchases = self.top_purchases.loc[self.top_purchases['user_id'] == user].head(N)
        
        recs = top_user_purchases['item_id'].apply(lambda x: self.id_to_itemid \
                                                   [self.model.similar_items(self.itemid_to_id[x], N=2)[1][0]]).tolist()
        
        recs = self._extend_with_top_popular(recs, N=N)
        
        assert len(recs) == N, 'Количество рекомендаций != {}'.format(N)
        return recs
    
    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        
        users = [self.id_to_userid[self.model.similar_users(self.userid_to_id[user], N=N+1)[i][0]] for i in range(1, N+1)]
        
        top_user_purchases = self.top_purchases.loc[self.top_purchases['user_id'].isin(users)].groupby('user_id', sort=False).head(1)
        recs = top_user_purchases['item_id'].unique().tolist()
        
        recs = self._extend_with_top_popular(recs, N=N)
        
        assert len(recs) == N, 'Количество рекомендаций != {}'.format(N)
        return recs
    
    def _get_recommendations(self, model, user, N=5):
        
        recs = [self.id_to_itemid[rec[0]] for rec in 
                    model.recommend(userid=self.userid_to_id[user], 
                                    user_items=csr_matrix(self.user_item_matrix),   # на вход user-item matrix
                                    N=N, 
                                    filter_already_liked_items=False, 
                                    filter_items=[self.itemid_to_id[999_999]], 
                                    recalculate_user=True)]
        
        recs = self._extend_with_top_popular(recs, N=N)
        
        return recs
    
    
    def get_als_recommendations(self, user, N=5):
        return self._get_recommendations(model=self.model, user=user, N=N)
    
    
    def get_own_recommendations(self, user, N=5):
        return self._get_recommendations(model=self.own_recommender, user=user, N=N)