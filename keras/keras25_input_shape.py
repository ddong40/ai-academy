#18_1에서 가져옴

import sklearn as sk
print(sk.__version__)   # 0.24.2
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

#1. 데이터
dataset = load_boston()
print(dataset)
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x)
print(x.shape)    
print(y)
print(y.shape)    

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7,
                                                    random_state=6666)
print('x_train :', x_train)
print('x_test :', x_test)
print('y_train :', y_train)
print('y_test :', y_test)

#2. 모델구성
model = Sequential()
# model.add(Dense(100, input_dim=13)) 
model.add(Dense(32, input_shape=(13,))) #백터형태로 받아들인다 백터가 어차피 column 이니까 #이미지일때는 input_shape=(8,8,1) 
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split = 0.3)
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)
print("걸린시간 : ", round(end - start, 2), "초" )
print('=====================hist==========')
print(hist)
print('======================= hist.history==================')
print(hist.history)
print('================loss=================')
print(hist.history['loss'])
print('=================val_loss==============')
print(hist.history['val_loss'])
print('====================================================')


# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
# plt.legend(loc='upper right') #라벨 값이 무엇인지 명시해주는 것이 레전드
# plt.title('Boston Loss') #그래프의 제목 
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()
# plt.show()