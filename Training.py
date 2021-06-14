import catboost
import numpy as np
import pandas as pd
from itertools import product
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import time
import sys
import argparse
import gc

class train():
    def __init__(self, DataName,modelname):
        self.DataName = DataName
        self.model = None
        self.data =None
        self.cat_feats = None
        self.modelname = modelname
        self.X_train = None
        self.Y_train = None
        self.X_valid = None
        self.Y_valid = None
        self.X_test = None

    def data_loader(self):
        self.data = pd.read_pickle(self.DataName)
        self.data = self.data[['date_block_num', 'shop_id', 'item_id', 'item_cnt_month', 'city',
       'item_category_id', 'main_type', 'sub_type', 'month', 'year',
       'item_shop_last_sale', 'item_last_sale', 'item_shop_first_sale',
       'item_first_sale']]
        self.cat_feats = ['shop_id','city','item_category_id','main_type','sub_type']

    def dataset_split(self,data):
        X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
        Y_train = data[data.date_block_num < 33]['item_cnt_month']
        X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
        Y_valid = data[data.date_block_num == 33]['item_cnt_month']
        X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
        return X_train,Y_train,X_valid,Y_valid,X_test


    def LGBM_model_building(self):
        model = LGBMRegressor(
            max_depth = 8,
            n_estimators = 500,
            colsample_bytree=0.7,
            min_child_weight = 300,
            reg_alpha = 0.1,
            reg_lambda = 1,
            random_state = 42,
        )

        model.fit(
            self.X_train, 
            self.Y_train, 
            eval_metric="rmse", 
            eval_set=[(self.X_train, self.Y_train), (self.X_valid, self.Y_valid)], 
            verbose=10, 
            early_stopping_rounds = 40,
            categorical_feature = self.cat_feats) # use LGBM's build-in categroical features.

        return model


    def XGB_model_building(self):
        model = XGBRegressor(
            max_depth=7,
            n_estimators=1000,
            min_child_weight=300,   
            colsample_bytree=0.8, 
            subsample=0.8, 
            gamma = 0.005,
            eta=0.1,    
            seed=42)

        model.fit(
            self.X_train, 
            self.Y_train, 
            eval_metric="rmse", 
            eval_set=[(self.X_train, self.Y_train), (self.X_valid, self.Y_valid)], 
            verbose=10, 
            early_stopping_rounds = 40,
            )
        return model

    def CatBoost_model_building(self):
        model = CatBoostRegressor(iterations=1000, loss_function='RMSE',
                            #   task_type="GPU",
                              learning_rate=0.06,  
                              depth=8,              
                              l2_leaf_reg=11,
                              random_seed=17, 
                              silent=True,
                              )


        model.fit( self.X_train, self.Y_train, 
                    cat_features=self.cat_feats,
                    early_stopping_rounds = 40,
                    verbose=10
                    )
        return model

    def test(self):
        Y_pred = self.model.predict(self.X_valid).clip(0, 20)
        Y_test = self.model.predict(self.X_test).clip(0, 20)

        X_train_level2 = pd.DataFrame({
            "ID": np.arange(Y_pred.shape[0]), 
            "item_cnt_month": Y_pred
        })

        submission = pd.DataFrame({
            "ID": np.arange(Y_test.shape[0]), 
            "item_cnt_month": Y_test
        })
        submission.to_csv('submission.csv', index=False)

    def main(self):
        self.data_loader()
        self.X_train,self.Y_train,self.X_valid,self.Y_valid,self.X_test = self.dataset_split(self.data)
        if self.modelname == 'LGBM':
            self.model = self.LGBM_model_building()
        elif self.modelname == 'XGB':
            self.model = self.XGB_model_building()
        elif self.modelname == 'CatBoost':
            self.model = self.CatBoost_model_building()

        self.test()




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre',
                        default='./data.pkl',
                        help='input data file name')
    
    args = parser.parse_args()
     
    Train = train(args.pre,'LGBM')
    # Train = train(args.pre,'XGB')
    # Train = train(args.pre,'CatBoost')
    Train.main()