import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import datetime
from chinese_calendar import is_workday
import lxml
import pandas as pd
import numpy as np
import requests
import json
from bs4 import BeautifulSoup
import re
import time

def create_assist_date(datestart = None,dateend = None):
    if datestart is None:
        datestart = '2016-01-01'
    if dateend is None:
        dateend = datetime.datetime.now().strftime('%Y-%m-%d')

    datestart=datetime.datetime.strptime(datestart,'%Y-%m-%d')
    dateend=datetime.datetime.strptime(dateend,'%Y-%m-%d')
    date_list = []
    date_list.append(datestart.strftime('%Y-%m-%d'))
    while datestart<dateend:
        datestart+=datetime.timedelta(days=+1)
        date_list.append(datestart.strftime('%Y-%m-%d'))
    return date_list


def is_work_day(date_str):
    j_date = date_str.split("-")
    j_date = datetime.date(int(j_date[0]), int(j_date[1]), int(j_date[2]))
    work_day = 0
    if is_workday(j_date):
       # 工作日
       work_day = 1
    else:
       # 休息日      
       work_day = 0
    return work_day

def create_assist_date_output(datestart = None,dateend = None):
    if datestart is None:
        datestart = '2016-01-01'
    if dateend is None:
        dateend = datetime.datetime.now().strftime('%Y-%m-%d')

    datestart=datetime.datetime.strptime(datestart,'%Y-%m-%d')
    dateend=datetime.datetime.strptime(dateend,'%Y-%m-%d')
    date_list = []
    date_list.append(datestart.strftime('%Y%m%d'))
    while datestart<dateend:
        datestart+=datetime.timedelta(days=+1)
        date_list.append(datestart.strftime('%Y%m%d'))
    print("data list length is ", len(date_list))
    return date_list

def get_newest_data():
    months = [3, 4]
    years = [2022]
    country = {'高雄': 59554, '屏東': 71301, '花蓮': 71305, '嘉義': 71298, '苗栗': 71302, '南投': 71304, "台北": 71294, '桃園': 71295, '新竹': 71295, '宜蘭': 71297, '台南': 71299, '台東': 71300, '雲林':71306, '彰化': 71303, '台中': 71082}
    index_ = ['date', 'MaxTemp', 'MinTemp', 'work']
    final = {}
    data = pd.DataFrame(columns=index_)
    new_data = pd.DataFrame()
    file_name = 0
    try:
        for idxx, k in enumerate(country.keys()):
            data = pd.DataFrame()
            for y in years:
              for m in months:
                url = 'https://tianqi.2345.com/Pc/GetHistory?areaInfo[areaId]='+ str(country[k])+'&areaInfo[areaType]=2&date[year]='+str(y)+'&date[month]='+str(m)
                response = requests.get(url=url)
                response.encoding='utf-8' 
                if response.status_code == 200:
                  html_str = json.loads(response.text)['data']
                  soup = BeautifulSoup(html_str, 'lxml')
                try:
                    tr = soup.table.select("tr")
                    for i in tr[1:]:
                      td  = i.select('td')
                      tmp = []
                      work_day = 1
                      for idx, j in enumerate(td[0:3]):
                         j = re.sub('<.*?>', "", str(j))
                         if idx == 0:
                           j = str(j).split(" ")
                           new_j = j[0]
                           work_day = is_work_day(new_j)
                         else:
                           new_j = str(j).replace("°", "")
                         tmp.append(new_j)
                         if idx == 2:
                           tmp.append(work_day)
                      data_spider = pd.DataFrame(tmp).T
                      data_spider.columns = index_
                      data = pd.concat((data, data_spider), axis=0)
                      time.sleep(5)
                      print("get data")
                except:
                     print("no data")
            if new_data.empty:
                new_data = data
            else:
                new_data = pd.merge(new_data, data, on='date')
            #print("new data is ", new_data)
        file_name = "tempture_data.csv"
        new_data = new_data.reset_index(drop=True)
        new_data.to_csv(file_name, encoding='utf_8_sig')
        print(file_name + " save success!")
        print("get newest done!")
    except:
        file_name = 0
        print("error to get newest weather data....")

    if file_name != 0:
        startDate = "2022-03-30"
        endDate = "2022-04-13"

        predict_date = create_assist_date(startDate, endDate)
        predict_date_dataframe = pd.DataFrame()
        predict_date_dataframe["date"] = predict_date


        predict_date = create_assist_date(startDate, endDate)
        predict_date_dataframe = pd.DataFrame()
        predict_date_dataframe["date"] = predict_date
        file_name = "tempture_data.csv" 
        tp_data = pd.read_csv(file_name)
        tp_df = pd.DataFrame(index=None)
        tp_df = pd.concat([tp_df,tp_data],ignore_index=True)
        #%%
        tp_df = pd.merge(tp_df, predict_date_dataframe, on='date')
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
        df_result["avg_max"] = df_max
        df_result["avg_min"] = df_min
        df_result["work_day"] = tp_df[4]

        # df_result為最終的檔案。

        #%%

        df_result.to_csv("fed_data.csv", header=True, index = False ,encoding='utf_8_sig')
    else:
        print("use backup weather")
        df_result = pd.read_csv("./data/backup_fed_data.csv")
    return df_result


if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='./data/training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.

    df_training = pd.read_csv(args.training)
    X = df_training.iloc[:,0:3]
    Y = df_training.iloc[:,3:4]

    minmax = preprocessing.MinMaxScaler()
    X_scaled = minmax.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size = 0.2, random_state = 0)
    regressor = make_pipeline(PolynomialFeatures(3), LinearRegression())
    regressor.fit(X_train, y_train)
    score = regressor.score(X_test, y_test)
    print("training done!")
    # print('Score: ', score)
    # print('Accuracy: ' + str(score*100) + '%')

    # 取得爬蟲的資料

    df_new_data = get_newest_data()

    # scale
    minmax = preprocessing.MinMaxScaler()
    df_new_data_scaled = minmax.fit_transform(df_new_data)
    # scale = StandardScaler()
    # df_new_data_scaled = pd.DataFrame(scale.fit_transform(df_new_data), columns=df_new_data.keys())

    datestart = "2022-03-30"
    dateend = "2022-04-13"

    date_list = create_assist_date_output(datestart, dateend)

    df_result = pd.DataFrame()
    result = regressor.predict(df_new_data_scaled)
    df_result["date"] = date_list
    final = []
    elec = 3400
    for data in result:
        data[0] = data[0]/10
        operating_reserve = elec - data[0]
        if operating_reserve < 0:
            final.append(287)
        elif operating_reserve < 150:
            final.append(287)
        elif operating_reserve > 700:
            final.append(500)
        else:
            final.append(elec-data[0])
        # final.append(data[0])
    df_result["operating_reserve(MW)"] = final
    df_result.to_csv(args.output, index=0)