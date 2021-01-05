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

data_=pd.read_csv("test_data.csv")
data_.columns = ["userId", "movieId", "rating", "timestamp"]
data1.columns = ["userId", "movieId", "rating"]

data1=data1[['userId', 'movieId', 'rating']]
data_test=data_[['userId', 'movieId', 'rating']]
print(data1.shape,data_test.shape)

algo = SVD()

reader=Reader()
d=Dataset.load_from_df(data1,reader=reader)

dataset,raw2inner_id_users,raw2inner_id_items= KFold().splitDate(d)
maes_name=[str(i) for i in range(300)]
maes=[str(i) for i in range(300)]
for u in range(300):

    user = []
    mae=[]

    algo.fit(dataset)

    for col in data_test["userId"].unique():

        data2 = data_test[data_test["userId"] == col]

        testdata= [(uid, iid, float(r)) for (uid, iid, r) in data2.itertuples(index=False)]

        predictions = algo.test(testdata)

        # 计算并打印RMSE

        b=accuracy.mae(predictions)

        print("用户是：",col,"，MAE:",b)
        user.append(col)
        mae.append(b)
    maes[u]=mae
    print(maes[u])

rate_insert = {
    'userId': user,

}

rate_insert = pd.DataFrame(rate_insert)

for i in range(300):
    rate_insert[maes_name[i]]=maes[i]
rate_insert.set_index('userId',drop=True,inplace=True)
rate_insert.to_csv("test\\"+str(2)+".csv")

maes=[str(i) for i in range(300)]
df1=pd.read_csv("test\\2.csv",index_col=[0])
print(df1)
df1["mean"]= df1[maes].mean(axis=1)

df2=df1["mean"]
print(df2)




