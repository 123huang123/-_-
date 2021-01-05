import pandas as pd
import os
# maes=[str(i) for i in range(300)]
# df1=pd.read_csv("test\\3.csv",index_col=[0])
# print(df1)
# df1["mean"]= df1[maes].mean(axis=1)
# df1=df1["mean"]
# df1.columns=["mae"]
# df1.to_csv('3_.csv')
#
# df2=pd.read_csv("test\\2.csv",index_col=[0])
# print(df2)
# df2["mean"]= df2[maes].mean(axis=1)
# df2=df2["mean"]
# df2.columns=["mae"]
# df2.to_csv('2_.csv')

# df1=pd.read_csv("1_2_合并.csv",index_col=[0])
# # print(df1)
# # df1["better"]=df1["mae_改进"]-df1["mae_初始"]
# # df1.to_csv("1_2_合并.csv")
# data_sort = df1.sort_values(by="better", axis=0,ascending=True)
# print(data_sort.head(100))
#
# print(list(data_sort.head(100).index.values))

data=pd.read_csv(r"train_data.csv")
data_user=data[data["userId"]==384]
print(len(data_user))