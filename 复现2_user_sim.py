import pandas as pd
import time
from surprise import Dataset
import os
from surprise import Reader, Dataset, accuracy, KNNBaseline, KNNWithMeans, KNNWithZScore, SVD
import pandas as pd
import random
import numpy as np
import math
from surprise import Reader, Dataset
from math import sqrt
from surprise.model_selection import KFold
from surprise.model_selection import ShuffleSplit
import collections
from sklearn.model_selection import train_test_split
import  time
with  open('./sim/%s' % ("item") + '.txt', "r") as ff:
    item_sim = eval(ff.read())

with  open('./sim/%s' % ("user") + '.txt', "r") as ff:
    user_sim = eval(ff.read())


#data=pd.read_csv("notInTopN.csv")
data1=pd.read_csv("train_data.csv")

rate_inserts =pd.DataFrame()
for user in data1["userId"].unique():
    data_user=data1[data1["userId"]==user]
    data_user_own_movies=list(data_user["movieId"])
    nei_user_dict = user_sim[user]



    nei_user=list(user_sim[user].keys())########该用户所有的邻居
    if len(nei_user)<1:
        continue
    data2 = data1[data1.userId.isin(nei_user)].reset_index(drop=True)
    a = list(data2["movieId"])  # 邻居们看过的电影
    b = collections.Counter(a)
    b = dict(b)
    dict2 = sorted(b.items(), key=lambda item: item[1], reverse=True)[0:20]  # 邻居共同观看的电影数量最多的前500个
    dict_b = {}
    for key, value in dict2:
        dict_b[key] = value
    print(user,"的邻居看过的次数最多的500部电影", list(dict_b.keys()))
    #user_not_top=list(data[data.userId==user]["movieId"])
    #print(user,"的notTop：",user_not_top)


    last_recommend_movies=[]
    for i in list(dict_b.keys()):#在邻居共同观看的电影数量最多的前20个
        if (i not in data_user_own_movies ):#如果用户没有看到，测试集里面也没有，就推荐给用户
            last_recommend_movies.append(i)
    print(user,"的最终推荐：",last_recommend_movies)



#找出全部电影邻居，进行计算预估评分
    A = []
    B = []
    for last_movie in last_recommend_movies:
        data2_last_movie=data2[data2.movieId==last_movie]
        list_last_movie_nei=list(data2_last_movie["userId"])
        last_movie_sim_sum=0
        last_movie_sum=0
        for list_last_movie_nei_user in list_last_movie_nei:
            last_movie_sim_sum+=nei_user_dict[list_last_movie_nei_user]
            b=list(data2_last_movie[data2_last_movie["movieId"] == last_movie]["rating"])[0]
            last_movie_sum+=nei_user_dict[list_last_movie_nei_user]*b
        rating=last_movie_sum/last_movie_sim_sum
        print(rating)
        A.append(last_movie)
        B.append(rating)

    rate_insert = {
        'movieId': A,
        "rating":B
    }
    rate_insert = pd.DataFrame(rate_insert)
    rate_insert["userId"]=user
    print(rate_insert)
    rate_inserts = rate_inserts.append(rate_insert, ignore_index=True)
rate_inserts=rate_inserts[["userId","movieId","rating"]]
print(rate_inserts)

rate_inserts.to_csv("train_data_1.csv",index=None)

a=pd.read_csv("train_data.csv")
del a["timestamp"]

b=pd.read_csv("train_data_1.csv")
print(a,b)
c=a.append(b,ignore_index=True)
c.to_csv("train_data_2.csv",index=None)



import numpy as np
import pandas as pd
import time
from  surprise import Dataset
import os
from surprise import Reader, Dataset,accuracy, KNNBaseline,KNNWithMeans,KNNWithZScore,SVD
import pandas as pd
import random
import numpy as np
import math
from math import sqrt
from sklearn.model_selection import train_test_split

from  surprise.model_selection import KFold
from  surprise.model_selection import ShuffleSplit
data1=pd.read_csv("train_data_2.csv")
data2=pd.read_csv("test_data.csv")
data2.columns = ["userId", "movieId", "rating", "timestamp"]
data1.columns = ["userId", "movieId", "rating"]

data1=data1[['userId', 'movieId', 'rating']]
data2=data2[['userId', 'movieId', 'rating']]
print(data1.shape,data2.shape)
A=[]
B=[]
for i in range(30):
    user_based_sim_option = {'name': 'cosine', 'user_based': True}#物品相似度
    for i in range(1):
        kf = KFold(n_splits=5)  # 定义交叉验证迭代器
        algo=SVD()

        reader=Reader()
        d=Dataset.load_from_df(data1,reader=reader)
        print("1")
        dataset,raw2inner_id_users,raw2inner_id_items= KFold().splitDate(d)
        print("2")
        algo.fit(dataset)
        print("3")
        testdata= [(uid, iid, float(r)) for (uid, iid, r) in data2.itertuples(index=False)]
        print("4")
        predictions = algo.test(testdata)
        print("5")
        # 计算并打印RMSE

        a=accuracy.rmse(predictions)
        b=accuracy.mae(predictions)
        A.append(a)
        B.append(b)
print("RMSE:",sum(A)/30)
print("MAE:",sum(B)/30)
