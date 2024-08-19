# 19일 월요일 종가를 맞춰봐
# 제한시간 18일 일요일 23시 59분
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Input, Concatenate, Bidirectional
from tensorflow.keras.models import Sequential, Model, load_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

path = 'C:/Users/ddong40/ai_2/_data/중간고사데이터/'

x_naver = pd.read_csv(path + 'NAVER 240816.csv', index_col=0,thousands=",")
x_hive = pd.read_csv(path + '하이브 240816.csv', index_col=0, thousands= ",")
y = pd.read_csv(path + '성우하이텍 240816.csv', index_col=0, thousands=",")

# print(x_naver.isnull().sum())
# print(x_hive.isnull().sum())
# print(y.isnull().sum())

# x_naver = x_naver.fillna(x_naver.mean())
# x_hive = x_hive.fillna(x_hive.mean())
# y = y.fillna(y.mean())

# print(x_naver.isnull().sum())
# print(x_hive.isnull().sum())
# print(y.isnull().sum())




x_naver = x_naver[:948]
x_hive = x_hive[:950]
y = y[:948]

x_naver = x_naver.sort_values(by=['일자'], ascending = True)
x_hive = x_hive.sort_values(by=['일자'], ascending = True)
y = y.sort_values(by=['일자'], ascending = True)

print(y)

x_naver = x_naver.drop(['전일비'], axis=1)
x_hive = x_hive.drop(['전일비'], axis=1)
x_naver = x_naver.drop(columns=x_naver.columns[4], axis=1)
x_hive = x_hive.drop(columns=x_hive.columns[4], axis=1)
y = y['종가']

x_naver = x_naver.astype(float)
x_hive = x_hive.astype(float)
y = y.astype(float)

print(y)
print(x_hive)
print(x_naver)


print(x_naver.shape) #(948, 14)
print(x_hive.shape) #(948, 14)
print(y.shape) #(948,)

x_naver_test = x_naver[-5:]
x_hive_test = x_hive[-5:]
y_test1 = y[5:]
x_naver = x_naver[:-5]
x_hive = x_hive[:-5]
print(x_naver_test)
print(y_test1)

size = 5

def split_x(dataset, size):
    aaa= []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array (aaa)

xxx_naver = split_x(x_naver, size)

xxx_hive = split_x(x_hive, size)

xxx_naver_test = split_x(x_naver_test, size)
xxx_hive_test = split_x(x_hive_test, size)

print(xxx_naver) #(939, 5, 14)
print(xxx_naver.shape) #(939, 5, 14)
print(xxx_hive.shape) #(939, 5)

yyy =split_x(y_test1, size)

print(yyy.shape) 
print(yyy)

yyy= yyy[:, 0]

print(yyy)
print(yyy.shape) #(939,)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(xxx_naver, xxx_hive, yyy, test_size=0.2, random_state=1024, shuffle=True)


model = load_model('./_save/중간고사가중치/keras63_99_성우하이텍_전사영.hdf5')

loss = model.evaluate([x1_test, x2_test], y_test)
y_predict = model.predict([xxx_naver_test, xxx_hive_test])

print('로스 : ', loss)
# print('시간 : ', round(end_time - start_time, 3), '초')
print('성우하이텍 8월 19일 종가 : ', y_predict[0][1], '원')

# 로스 :  3194060.0
# 성우하이텍 8월 19일 종가 :  7482.5493 원