# 2021-DSAI-HW4作業說明
## Report link：
https://drive.google.com/file/d/1SrrT1Bgcrmwspo7bkn3fuZRRx1kMi2TC/view?usp=sharing

## 使用說明:

安裝相依套件
```
pip install -r requirements.txt
```

模型訓練
```
python Training.py
```

資料前處理
```
python DataPre-processing.py
```

## 檔案說明:

* `DataPre-processing.py` : 關於資料前處理(包含資料清洗、特徵製作與測試資料處理)的程式碼，運用data資料夾內部的檔案來進行製作，最後把特徵保存到`data.pkl`。
* `Training.py` : 模型訓練與競賽預測，預設為`Lightgbm`的模型訓練，可以透過更改`if __name__ == '__main__'`來更改想要訓練的模型(包含`CatBoost`與`XGBoost`)，訓練結束後會進行競賽預測並且輸出`submission.py`，此檔案為競賽預測繳交檔案。
* `data.pkl` : 保存特徵的檔案。(檔案過大請另外下載：[連結](https://drive.google.com/file/d/1T22uUYuXPiemTi9ZY9sXHOgSxoSoXLRF/view?usp=sharing))
* `submission.csv` : 為競賽最終所繳交的檔案。
* data資料夾文件說明(檔案過大請另外下載：[連結](https://drive.google.com/drive/folders/1T3xXpwRPw26YVaQN80_ATuBQQKMt0KHS?usp=sharing))
    * `sales_train.csv` - the training set. Daily historical data from January 2013 to October 2015.
    * `test.csv` - the test set. You need to forecast the sales for these shops and products for November 2015.
    * `sample_submission.csv` - a sample submission file in the correct format.
    * `items.csv` - supplemental information about the items/products.
    * `item_categories.csv`  - supplemental information about the items categories.
    * `shops.csv` - supplemental information about the shops.

## 資料分析:

每個月的平均銷售量：
* 每年的12月銷售量特別突出
* 每年銷售量有週期性  
![](https://i.imgur.com/IeP1iiu.png)

## 資料清洗:

透過刪除離群值讓模型擬合過程更順利

異常的銷售量  
![](https://i.imgur.com/eeSgCz5.png)


異常的商品價格  
 ![](https://i.imgur.com/2SqST7Z.png)


## 特徵選擇:
* item_category_id : 商品類別ID
* data_black_num : 時間編號
* shop_id : 商店ID
* Item_id : 商品ID
* city : 每間商店位於的城市
* main_type : 商品的主要類別(例如:遊戲機、配件)
* sub_type : 商品的子類別(例如:PS4、XBOX)
* month : 月份
* year : 年(1或2)
* Item_shop_last_sales : 從商品與商店的組合中找出最後一次銷售的時間
* Item_last_sales : 商品最後一次銷售的時間
* Item_shop_first_sales : 從商品與商店的組合中找出第一次銷售的時間
* Item_first_sales : 商品第一次銷售的時間

## 模型選擇:
採用Kaggle競賽常用的模型，包含`XGBoost`、`LightGBM`、`CatBoost`。

## 訓練結果:
此競賽採用RMSE來做為評分標準

以下分別測試了三種模型得出的結果：  
![](https://i.imgur.com/0s0LBBS.png)

重要特徵：  
![](https://i.imgur.com/TXNj22z.png)