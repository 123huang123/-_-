import numpy as np
import pandas as pd
import time
import numpy as np
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

from sklearn.model_selection import train_test_split
import  time
u_data=pd.read_csv(r"train_data.csv")
data = pd.DataFrame(u_data)
data.columns = ["userId", "movieId", "rating", "timestamp"]

data_sort = data.sort_values(by="timestamp", axis=0)
print(data_sort)


A=pd.DataFrame()


data = data[["userId", "movieId", "rating"]]

user_based_sim_option = {'name': 'cosine', 'user_based': True}
reader = Reader()
algo = KNNBaseline(sim_options=user_based_sim_option)
d1 = Dataset.load_from_df(data, reader=reader)
dataset, raw2inner_id_users,raw2inner_id_items = KFold().splitDate(d1)
# f1 = {value: key for key, value in raw2inner_id_users.items()}
# f2 = {value: key for key, value in raw2inner_id_items.items()}
# qw = open('id_change\\userid_to_real.txt','w')
# qw.write(str(f1))
# qw.close()
# qw2 = open('id_change\\itemid_to_real.txt','w')
# qw2.write(str(f2))
# qw2.close()
m = raw2inner_id_users
f = {value: key for key, value in m.items()}
# 训练并测试算法,建立模型
w = algo.fit(dataset)
s = w.sim
get_neighbors = {}
for col in data["userId"].unique() :
    dict2 = {}
    c=[]
    A2 = m[col]  # col用户对于评分缺失物品的内部id
    w1 = w.get_neighbors(A2, 20)  # 此物品100个邻居
    for r in w1:
        sim_score=s[r][A2]
        if sim_score>0:
            c.append(f[r])
            print("{0}和{1}的相似度为{2}".format(col,f[r],sim_score))
            dict2[f[r]]=sim_score
    print(col,c)
    get_neighbors[col] = dict2
print(get_neighbors)
f=open('./sim/%s' %("user") + '.txt', "w")
f.write(str(get_neighbors))
f.close()

