#https://www.kaggle.com/competitions/otto-group-product-classification-challenge/data

#0.89점 이상 뽑아내기 ㅎㅎ


import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


#1 데이터
path = 'C:/Users/ddong40/ai_2/_data/kaggle/otto-group-product-classification-challenge/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
sampleSubmission = pd.read_csv(path + 'sampleSubmission.csv', index_col = 0)

x = train_csv.drop('target', axis=1)
y = train_csv['target']

scaler = LabelEncoder()
y = scaler.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1234)

from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import xgboost as xgb

#kfold
parameters = [
    {'n_jobs' : [-1], 'n_estimators' : [100, 500], 'max_depth' : [6, 10 ,12],
     'min_samples_leaf' : [3, 10], 'learning_rate' : [0.0001, 0.001, 0.01]}, #36
    {'n_jobs' : [-1], 'max_depth' : [6, 8,  10 ,12],
     'min_samples_leaf' : [3, 5, 7, 10], 'learning_rate' : [0.0001, 0.001, 0.01]}, #48
    {'n_jobs' : [-1], 'min_samples_leaf' : [3, 5, 7, 10], 
     'min_samples_leaf' : [2, 3, 5, 10], 'learning_rate' : [0.0001, 0.001, 0.01]}, #48
    {'n_jobs' : [-1], 'min_samples_leaf' : [2, 3, 5, 10], 'learning_rate' : [0.0001, 0.001, 0.01]}, #12
] #134

from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
import xgboost as xgb

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=3333)


model = RandomizedSearchCV(xgb.XGBClassifier(device = 'cuda:0'), parameters, cv=kfold,
                     verbose=True,
                     refit=True,
                     n_jobs=-1, 
                     n_iter= 9,
                     
                     ) 
start_time = time.time()

model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = True)

end_time = time.time()



print('최적의 매개변수 : ', model.best_estimator_) 

print('최적의 파라미터 : ', model.best_params_) 
# 최적의 파라미터 :  {'n_jobs': -1, 'n_estimators': 500, 'min_samples_leaf': 3, 'max_depth': 12, 'learning_rate': 0.01}
print('최고의 점수 : ', model.best_score_) 
# 최고의 점수 :  0.8033816459718894
print('모델의 점수 : ', model.score(x_test, y_test)) 
# 모델의 점수 :  0.8158532643826761
y_predict = model.predict(x_test)

print('accuracy_score : ', accuracy_score(y_test, y_predict)) 
# accuracy_score :  0.8158532643826761
y_pred_best = model.best_estimator_.predict(x_test) 

print('최적 튠 ACC: ', accuracy_score(y_test, y_pred_best)) 
# 최적 튠 ACC:  0.8158532643826761
print('걸린시간 : ', round(end_time - start_time, 2), '초') 
# 걸린시간 :  689.65 초
