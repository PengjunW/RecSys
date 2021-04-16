import numpy as np
import pandas as pd
from pprint import pprint

users=['u1','u2','u3','u4','u5']
items=['item1','item2','item3','item4','item5']

datasets=[
    [5,3,4,4,None],
    [3,1,2,3,3],
    [4,3,4,3,5],
    [3,3,1,5,4],
    [1,5,5,2,1]
]
df=pd.DataFrame(datasets,columns=items,index=users)

# User_based
#默认是计算列的相似度，所以要转置
user_sim=df.T.corr()
#pprint(user_sim)

topN_user={}
for i in user_sim.index:
    _df=user_sim.loc[i].drop([i])
    _df_sorted=_df.sort_values(ascending=False)
    top2=list(_df_sorted.index[:2])
    topN_user[i]=top2
#pprint(topN_user)

# 预测用户1的item5的得分是多少
sim_u1=topN_user['u1']
sim_u_score=0.
sim_u=0.
for i in sim_u1:
    sim_u+=user_sim.loc['u1',i]
    sim_u_score+=df.loc[i,"item5"]*user_sim.loc['u1',i]
print(sim_u_score/sim_u)

# Item_based
item_sim=df.corr()
#pprint(item_sim)

topN_item={}
for i in item_sim:
    _df=item_sim.loc[i].drop([i])
    _df_sorted=_df.sort_values(ascending=False)

    top2=list(_df_sorted.index[:2])
    topN_item[i]=top2
#pprint(topN_item)

# 预测用户1的item5的得分是多少
sim_item=topN_item['item5']
sim_i=0.
sim_i_score=0.
for i in sim_item:
    sim_i+=item_sim.loc['item5',i]
    sim_i_score+=df.loc['u1',i]*item_sim.loc['item5',i]
print(sim_i_score/sim_i)


