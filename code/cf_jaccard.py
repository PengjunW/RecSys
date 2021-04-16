import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.metrics.pairwise import pairwise_distances

users=['u1','u2','u3','u4','u5']
items=['item1','item2','item3','item4','item5']

datasets=[
    [1,0,1,1,0],
    [1,0,0,1,1],
    [1,0,1,0,0],
    [0,1,0,1,1],
    [1,1,1,0,1]
]
df=pd.DataFrame(datasets,columns=items,index=users)

# User_based CF
#计算杰卡德相关系数
users_sim=1-pairwise_distances(df.values,metric='jaccard')
users_sim=pd.DataFrame(users_sim,columns=users,index=users)
#pprint(users_sim)

#取出top2的相关用户
topN_user={}
for i in users_sim.index:
    _df=users_sim.loc[i].drop([i])
    _df_sorted=_df.sort_values(ascending=False)

    top2=list(_df_sorted.index[:2])
    topN_user[i]=top2
#pprint(topN_user)

#过滤
rs_user_based={}
for user,sim_users in topN_user.items():
    rs_result=set()
    for sim_user in sim_users:
        #print('sim_users ', sim_users)
        #print('sim_users items :', df.loc[sim_user])
        #print('df loc :', df.loc[sim_user].replace(0, np.nan).dropna().index)
        rs_result=rs_result.union(set(df.loc[sim_user].replace(0,np.nan).dropna().index))
        rs_result-=set(df.loc[user].replace(0,np.nan).dropna().index)
        rs_user_based[user]=rs_result

print("UserBased CF：")
pprint(rs_user_based)

#Item_based CF
item_sim=1-pairwise_distances(df.T.values,metric="jaccard")
item_sim=pd.DataFrame(item_sim,columns=items,index=items)
#pprint(item_sim)

#找出TopN相似的物品
topN_item={}
for i in item_sim.index:
    _df=item_sim.loc[i].drop([i])
    _df_sorted=_df.sort_values(ascending=False)
    top2=list(_df_sorted.index[:2])
    topN_item[i]=top2
#pprint(topN_item)

rs_item_based={}
for user in df.index:
    rs_item=set()
    for item in df.loc[user].replace(0,np.nan).dropna().index:
        rs_item=rs_item.union(topN_item[item])
    rs_item-=set(df.loc[user].replace(0,np.nan).dropna().index)
    rs_item_based[user]=rs_item
print("ItemBased CF：")
pprint(rs_item_based)
