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
data1=pd.read_csv("train_data.csv")
data2=pd.read_csv("test_data.csv")
data2.columns = ["userId", "movieId", "rating", "timestamp"]
data1.columns = ["userId", "movieId", "rating", "timestamp"]

data1=data1[['userId', 'movieId', 'rating']]
data2=data2[['userId', 'movieId', 'rating']]
print(data1.shape,data2.shape)

A=[]
B=[]
for i in range(30):

    for i in range(1):
        algo = SVD()
        d=Dataset.load_from_df(data1,reader=Reader())
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