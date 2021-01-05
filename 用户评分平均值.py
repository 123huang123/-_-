import pandas as pd
u_data=pd.read_csv(r"train_data.csv")
a={}
for movie in u_data["movieId"].unique() :
    a[movie]=u_data[u_data["movieId"]==movie]["rating"].mean()
print(a)

qw = open('电影评分平均值\\movie_mean.txt','w')
qw.write(str(a))
qw.close()