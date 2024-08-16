import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_wine
import time

#1 데이터
datasets = load_wine()

x = datasets.data
y = datasets.target

print(datasets)
print(datasets.DESCR)

from sklearn.preprocessing import OneHotEncoder
y = y.reshape(-1,1)
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, shuffle= True, 
                                                    random_state= 150, stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape) #124, 13
print(x_test.shape) #54, 13

x_train = x_train.reshape(124, 13, 1)
x_test = x_test.reshape(54, 13, 1)


#2 모델구성

model = Sequential()
model.add(LSTM(10, input_shape=(13, 1)))
model.add(Dense(128, 'relu'))
model.add(Dense(128, 'relu'))
model.add(Dense(128, 'relu'))
model.add(Dense(128, 'relu'))
model.add(Dense(128, 'relu'))
model.add(Dense(64, 'relu'))
model.add(Dense(64, 'relu'))
model.add(Dense(64, 'relu'))
model.add(Dense(64, 'relu'))
model.add(Dense(64, 'relu'))
model.add(Dense(3, 'softmax'))

#컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 50,
    restore_best_weights= True)

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path1 = './_save/keras59/09_wine/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path1, 'k30_', date, '_', filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose =1,
    save_best_only=True,
    filepath = filepath
)


model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1, validation_split=0.25, callbacks=[es, mcp])
end_time = time.time()

#평가 예측
loss = model.evaluate(x_test, y_test)
print("로스값 : ", loss[0])
print("accuracy : ", round(loss[1],3))
print("걸린시간 : ", round(end_time - start_time, 2), "초" )
y_pred = model.predict(x_test)
print(y_pred)

# 로스값 :  0.2663238048553467
# accuracy :  0.907

# minmaxscaler
# 로스값 :  0.2635650038719177
# accuracy :  0.963

# standardscaler
# 로스값 :  0.21828098595142365
# accuracy :  0.981

# MaxAbsScaler
# 로스값 :  0.9603663682937622
# accuracy :  0.944

# RobustScaler
# 로스값 :  0.23232993483543396
# accuracy :  0.963

# 세이브 값
# 로스값 :  0.3106008470058441
# accuracy :  0.963

# LSTM
# 로스값 :  0.8540423512458801
# accuracy :  0.87
# 걸린시간 :  71.64 초