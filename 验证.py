import  pandas as pd
u_data=pd.read_csv(r"train_data.csv")
u_data["timestamp"] = (u_data["timestamp"]) / (86400)
u_data1=u_data[u_data["userId"]==363]
print(u_data1)
u_data2=u_data[u_data["userId"]==335]
print(u_data2)

f = open('sim\\user.txt', 'r')
b = f.read()
dict_name_user = eval(b)
f.close()
print(dict_name_user[363])
