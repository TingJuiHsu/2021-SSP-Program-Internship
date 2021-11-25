import numpy as np
import pandas as pd
import xgboost
from xgboost import *
from xgboost.sklearn import XGBClassifier
import pickle
import gzip
import time
from pre import Data_Generation

from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split


## XGBOOST training, by Ting-Jui Hsu, NTHU, Taiwan, 2021/11/25

## load data 
#data = np.load('bof_data/des_2_data/data.npz')
data = np.load("training_data/data.npz")

X_train = data['X_train']
Y_train = data['Y_train']
X_test = data['X_test']
Y_test = data['Y_test']

# Or use the function Data_Generation() to load data
#X_train, Y_train, X_test, Y_test, L= Data_Generation()

print(X_train.shape)
print(Y_train.shape)

X_train, X_test = X_train/255, X_test/255
X_train = X_train.reshape(X_train.shape[0],30*128*3)
X_test = X_test.reshape(X_test.shape[0],30*128*3)

## hyper parameters 
hp = {
    'learning_rate': 0.1,
    'objective': 'multi:softmax',
    'num_class': 2,
    'n_estimators': 100,
    'max_depth': 3,
    #'subsample':0.8,
    #'colsample_btree':0.8,
    #'reg_lambda': 1,
    'tree_method': 'gpu_hist',
    'eval_metric': 'auc',
     #'min_child_weight' : 1,
}

## load model
with gzip.open('model/test_04_model.pgz', 'r') as f:
    xgbc = pickle.load(f)

i_t = time.time()
## Or initialize the model
#xgbc = XGBClassifier(**hp)#(**hp)

print(xgbc)

## training model
#xgbc.fit(X_train,Y_train, early_stopping_rounds=10, eval_set=[(X_test, Y_test)],verbose=True)
#xgbc.fit(X_train, Y_train, eval_set=[(X_test, Y_test)])

print(" Cost time: %.3f (sec)" %(time.time() - i_t))

## testing model
test_on_train_data = xgbc.score(X_train,Y_train)
test_on_test_data = xgbc.score(X_test,Y_test)

print("Score on train data: ",test_on_train_data)

print("Score on test data: ",test_on_test_data)

## save model
#with gzip.GzipFile('11_23/test_04_model.pgz', 'w') as f: ## reslut4 on ppt
#    pickle.dump(xgbc, f)







'''
xgbc = XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric='auc', gamma=0, gpu_id=0, importance_type=None,
              interaction_constraints='', learning_rate=0.2, max_delta_step=0,
              max_depth=3, min_child_weight=1, n_estimators=30, n_jobs=12,
              num_class=2, num_parallel_tree=1, objective='multi:softmax',
              predictor='auto', random_state=33, reg_alpha=0, reg_lambda=2,
              scale_pos_weight=None, subsample=1)
'''
'''
hp = {
    'learning_rate': 0.01,
    #'n_estimators':1000,
    #'objective': 'reg:squarederror',
    #'objective': 'binary:logistic',
    'objective': 'multi:softmax',
    'num_class': 2,
    'n_estimators': 1000,
    'max_depth': 3,
    'subsample':0.8,
    'colsample_btree':0.8,
    #'alpha': 0.8,
    #'reg_lambda': 1,
    #'booster':'gblinear',
    #'gamma':0.2,
    'tree_method': 'gpu_hist',
    #'random_state': 42, 
    'eval_metric': 'auc',
     #'min_child_weight' : 1,
}
'''