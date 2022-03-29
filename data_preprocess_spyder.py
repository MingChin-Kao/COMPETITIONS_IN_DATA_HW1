# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a preprocess data file for training data
"""
#%%

import pandas as pd

#%%

power_file = "./data/台灣電力公司_過去電力供需資訊.csv"
power_data = pd.read_csv(power_file)

#%%
# 日期轉換 20200222 轉成 2022-02-22
for idx, i in enumerate(power_data["日期"]):
    new_str = str(i)[:4] + '-' + str(i)[4:6] + "-" + str(i)[6:]
    power_data["日期"][idx] = new_str
    
#%%
# 溫度資料
tpp_file = "./各地區/tp_data.csv"
tpp_data = pd.read_csv(tpp_file)

#%%

power_data["date"] = power_data["日期"]
nn_pd = pd.DataFrame()
nn_pd = pd.merge(power_data, tpp_data, on='date')


#%%

tp_file = "./各地區/tp_data.csv"
tp_data = pd.read_csv(tp_file)
tp_df = pd.DataFrame(index=None)
tp_df = pd.concat([tp_df,tp_data],ignore_index=True)
tp_df.columns = range(tp_df.shape[1])

#%%

df_max = pd.DataFrame()
df_min = pd.DataFrame()
for i in range(int((len(tp_df.columns)-2)/3)):
    print(i)
    if i == 0:
        df_max = tp_df[i*3 + 2]
        df_min = tp_df[i*3 + 3]
    else:
        df_max = df_max + tp_df[i*3 + 2]
        df_min = df_min + tp_df[i*3 + 3]

#%%
df_max = df_max / 15
df_min = df_min / 15

#%%

# 將data整理成一個dataframe
df_result = pd.DataFrame()
df_result["avg_max"] = df_max[0:424]
df_result["avg_min"] = df_min[0:424]
df_result["work_day"] = tp_df[4][0:424]
df_result["power"] = nn_pd["尖峰負載(MW)"]

# df_result為最終的檔案。

#%%

df_result.to_csv("training_data.csv", header=True, index = False ,encoding='utf_8_sig')

#%% test

file = "./training_data.csv"
ddata = pd.read_csv(file)


#%%

import numpy as np
## 導入視覺化套件
import matplotlib.pyplot as plt
## 導入Sklearn套件
## 導入將數據集拆成訓練集與測試集的套件
from sklearn.model_selection import train_test_split
## 導入迴歸模型套件
from sklearn.linear_model import LinearRegression
## 導入多項式套件，建構多項式迴歸模型所需的套件
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#%%

X = ddata.iloc[:,0:3]
Y = ddata.iloc[:,3:4]

#%%

scale = StandardScaler() #z-scaler物件
X_scaled = pd.DataFrame(scale.fit_transform(X),
                                columns=X.keys())

#%%

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size = 0.1, random_state = 0)

#%%

regressor = make_pipeline(PolynomialFeatures(3), LinearRegression())
regressor.fit(X_train, y_train)


#%%
score = regressor.score(X_test, y_test)
print('Score: ', score)
print('Accuracy: ' + str(score*100) + '%')

#%%

import pickle

#%%

filename = "model.sav"
pickle.dump(regressor, open("./model/" + filename, 'wb'))
    
#%%

model2 = pickle.load(open('./model/model.sav', 'rb'))
#%%

a = model2.predict(X_test)


#%%

import datetime
#%%

def create_assist_date(datestart = None,dateend = None):
    if datestart is None:
        datestart = '2016-01-01'
    if dateend is None:
        dateend = datetime.datetime.now().strftime('%Y-%m-%d')

    # 转为日期格式
    datestart=datetime.datetime.strptime(datestart,'%Y-%m-%d')
    dateend=datetime.datetime.strptime(dateend,'%Y-%m-%d')
    date_list = []
    date_list.append(datestart.strftime('%Y-%m-%d'))
    while datestart<dateend:
        datestart+=datetime.timedelta(days=+1)
        date_list.append(datestart.strftime('%Y%m%d'))
    print(date_list)

#%%
datestart = "2022-03-25"
dateend = "2022-04-05"
date_list = create_assist_date(datestart, dateend)