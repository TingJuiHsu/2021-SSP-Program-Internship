import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import *
#from xgboost import XGBClassifier
import pickle
import gzip
import time

## XGBOOST training, by Ting-Jui Hsu, NTHU, Taiwan, 2021/12/03

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets  import  make_hastie_10_2
from xgboost.sklearn import XGBClassifier

## load data
data = np.load("w_o_aug/4data.npz")
#data = np.load("w_aug/4data_aug.npz")

X_train = data['X_train']
Y_train = data['Y_train']
X_test = data['X_test']
Y_test = data['Y_test']

# Or use the function Data_Generation() to load data
#X_train, Y_train, X_test, Y_test, L= Data_Generation()

print(X_train.shape)
print(Y_train.shape)
#print(Y_test[0:200])
print(X_test.shape)
print(Y_test.shape)

X_train, X_test = X_train/255, X_test/255
X_train = X_train.reshape(X_train.shape[0],30*128*3)
X_test = X_test.reshape(X_test.shape[0],30*128*3)

## data transform
xg_train = xgb.DMatrix(X_train, label=Y_train)
xg_test = xgb.DMatrix(X_test, label=Y_test)

param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'

param['eta'] = 0.05
param['max_depth'] = 4
#param['silent'] = 1
#param['nthread'] = 4
#param['eval_metric'] = 'auc'
param['num_class'] = 4
param['subsample'] = 0.8
param['tree_method'] = 'gpu_hist'
#param['min_child_weight'] = 2
#param['gamma'] = 2
param['max_delta_step'] = 5
param['scale_pos_weight'] = 1

watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 1000

## load model
with gzip.open('t12_01_model11.pgz', 'r') as f:
    bst = pickle.load(f)


## training
#bst = xgb.train(param, xg_train, num_round, watchlist )

## testing
pred_train = bst.predict( xg_train )##
pred_test = bst.predict( xg_test )##

from sklearn import metrics
print ('ACC: %.4f' % metrics.accuracy_score(Y_train,pred_train))
print(metrics.confusion_matrix(Y_train,pred_train))

print ('ACC: %.4f' % metrics.accuracy_score(Y_test,pred_test))
print(metrics.confusion_matrix(Y_test,pred_test))

## save model
#with gzip.GzipFile('11_18_model/test_06_model.pgz', 'w') as f:
#    pickle.dump(bst, f)
