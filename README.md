# COMPETITIONS_IN_DATA_HW1

### 預測目標

> 備轉容量(perating Reserve) = 系統運轉淨尖峰能力－系統瞬時尖峰負載(瞬間值)。
> 

透過台電網站觀察 [https://www.taipower.com.tw/d006/loadGraph/loadGraph/load_reserve_.html](https://www.taipower.com.tw/d006/loadGraph/loadGraph/load_reserve_.html) ，系統運轉尖峰能力在前半年的時間大約會落在 3000~3500(萬瓩)，假定為一定值，故本模型將以預測系統瞬時尖峰負載為目標。透過系統運轉尖峰能力(定值)減掉預測系統尖峰負載，最後加上一些限制，作為最後的輸出答案。

![image](https://github.com/MingChin-Kao/COMPETITIONS_IN_DATA_HW1/blob/main/elec.png)

### 資料集、資料前處理

- 資料集時間：20210101-20220327
    - 平均高溫、平均低溫
        
        利用爬從程式碼從 [https://tianqi.2345.com/wea_history/71297.htm](https://tianqi.2345.com/wea_history/71297.htm) 網站爬取2021年到2022年2越，台灣各地每天的氣溫，並將以日為單位，計算出每日的平均高溫及平均低溫。
        
    - 是否為六日及國定假日
        
        透過python套件`chinese_calendar` 查詢該日期是否有放假，若是無放假標示1，反之標示0
        
    - 將上述資料整理成
        
        
        | date | MaxTemp | MinTemp | is_workday | Operating Reserve |
        | --- | --- | --- | --- | --- |

### 訓練模型

將資料整理完之後，將會使用polynomial regression來對資料進行分析及預測。

將訓練資料及測試資料以8:2進行切割

### 預測

- 利用爬蟲程式碼抓取20220330~20220413的天氣資料當作模型的input資料。
- 經測試，發現網站有限制同個ip的存取次數，若短時間內存取過多次，則會造成錯誤。程式碼中也適當加上了sleep，避免短時間存取多次被阻擋，**所以在抓資料這部分會花比較多的時間**。為了避免爬蟲程式的錯誤導致無法順利產生submission.csv檔案，製作了一份backup_fed_data.csv檔案，裡面為20220315~20220328的天氣資料，若無法順利爬到最新的天氣資訊，則用最近的資料進行取代輸入模型中。
- 系統運轉尖峰能力在本程式碼中設定為3400(萬瓩)，利用模型預測出尖峰負載後將會進行相減，並針對過於大及過於小的值進行校正。根據網站的觀察，備轉容量並不會低於0，最低大約落在200上下，故下限我們將設定為287，若相減低於150，則調整成287，若上限超過700，則調整為500，根據觀察，除了上一次全台大停電以外，備轉容量皆沒有高於700。

### How to Run

```
python app.py --training ./data/training_data.csv --output submission.csv
```
