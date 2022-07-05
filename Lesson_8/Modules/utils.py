import pandas as pd
import numpy as np


def prefilter_items(data, item_features=None, n_popular=5000):
    
#     # Уберем самые популярные товары (их и так купят)
    
#     popularity = data.groupby('item_id', sort=False)['quantity'].sum().reset_index()
#     top_100 = popularity.sort_values('quantity', ascending=False).head(100)['item_id'].tolist()
#     data = data.loc[~data['item_id'].isin(top_100)]
    
    # Уберем товары, которые не продавались за последние 12 месяцев
    
    week_min = data['week_no'].max() - 52
    recency = data.groupby('item_id', sort=False)['week_no'].max().reset_index()
    recent = recency.loc[recency['week_no'] >= week_min, 'item_id'].tolist()
    data = data.loc[data['item_id'].isin(recent)]
    
    # Уберем неинтересные для рекомендаций категории (department)
    # Это те категории, где товаров меньше 20, а также 'MISC. TRANS.', 'MISC SALES TRAN'
    
    if item_features is not None:
        unwanted = item_features.groupby('department', sort=False)['item_id'].nunique().\
                loc[item_features.groupby('department', sort=False)['item_id'].nunique() <=20].index.tolist()
        unwanted += ['MISC. TRANS.', 'MISC SALES TRAN']
        proper_items = item_features.loc[~item_features['department'].isin(unwanted), 'item_id'].unique().tolist()
        data = data.loc[data['item_id'].isin(proper_items)]
    
    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.

    prices = data.groupby('item_id', sort=False)['sales_value'].sum() / \
             np.clip(data.groupby('item_id', sort=False)['quantity'].sum(), a_min=1, a_max=None)
    
    prices = prices.reset_index(name='price')
    
    too_cheap = prices.loc[prices['price'] <= 1, 'item_id'].tolist()
    
    data = data.loc[~data['item_id'].isin(too_cheap)]
    
    # Уберем слишком дорогие товары
    
    too_expensive = prices.loc[prices['price'] >= 50, 'item_id'].tolist()
    
    # Оставим только n самых популярных товаров
    
    popular = data.groupby('item_id', sort=False)['quantity'].sum().reset_index()
    top = popular.sort_values('quantity', ascending=False).head(n_popular).item_id.tolist()
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999_999
    
    return data
    
def postfilter_items(user_id, recommednations):
    pass