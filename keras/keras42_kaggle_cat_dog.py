# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

train_datagen = ImageDataGenerator(rescale=1./255,
    horizontal_flip=True, 
    vertical_flip=True, 
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    rotation_range=5, 
    zoom_range=1.2, 
    shear_range=0.7, 
    fill_mode='nearest'   
)
test_datagen = ImageDataGenerator(
    rescale=1./255
)



path_train = 'C:/Users/ddong40/ai_2/_data/kaggle/dogs-vs-cats-redux-kernels-edition/train/'
path_test = 'C:/Users/ddong40/ai_2/_data/kaggle/dogs-vs-cats-redux-kernels-edition/test/'
path = 'C:/Users/ddong40/ai_2/_save/keras42/cat_dog/'
path2 = 'C:/Users/ddong40/ai_2/_data/kaggle/dogs-vs-cats-redux-kernels-edition/'

sampleSubmission = pd.read_csv(path2 + 'sample_submission.csv', index_col = 0)

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(80,80),
    batch_size=25000,
    class_mode='binary',
    color_mode = 'rgb',
    shuffle = True    
)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size= (80,80),
    batch_size=12500,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False
)

x = xy_train[0][0]
y = xy_train[0][1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

x_test1 = xy_test[0][0]


print(x_train.shape) #(25000, 100, 100, 3)
print(y_train.shape) #(25000, )
print(y_train)

#모델 
# model = Sequential()
# input1 = Input(shape=(80, 80, 3))
# conv1 = Conv2D(32, 2, activation='relu', padding='same')(input1)
# maxpool = MaxPooling2D()(conv1)
# drop1 = Dropout(0.2)(maxpool)
# conv2 = Conv2D(32, 2, activation='relu', padding='same')(drop1)
# drop2 = Dropout(0.2)(conv2)
# conv3 = Conv2D(32, 2, activation='relu', padding='same')(drop2)
# drop3 = Dropout(0.2)(conv3)
# conv4 = Conv2D(32, 2, activation='relu', padding='same')(drop3)
# flatten = Flatten()(conv4)
# dense1 = Dense(32, activation='relu')(flatten)
# drop4 = Dropout(0.2)(dense1)
# dense2 = Dense(32, activation='relu')(drop4)
# drop5 = Dropout(0.2)(dense2)
# dense3 = Dense(32, activation='relu')(drop5)
# drop6 = Dropout(0.2)(dense3)
# dense4 = Dense(16, activation='relu')(drop6)
# output1 = Dense(1, activation='sigmoid')(dense4)
# model = Model(inputs = input1, outputs = output1)

model = Sequential()
model.add(Conv2D(32, 2, input_shape=(80, 80, 3), padding='same'))
model.add(MaxPooling2D())
model.add(Randomzoom(0.2))
model.add(RandomFlip('horizontal_and_vertical'))
model.add(RandomContrast(factor=0.90))
model.add(BatchNormalizotion())
model.add(Conv2D(32, 2, activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, 2, activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, 2, activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='softmax'))

#컴파일 훈련

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 30,
    restore_best_weights=True,
    verbose=1
)

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = './_save/keras42/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path1, 'k30_', date, '_', filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
    verbose=1,
    filepath=filepath
)

model.fit(x_train, y_train, epochs= 500, batch_size=8, verbose=1, validation_split=0.25, callbacks=[es, mcp])
end_time = time.time()

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_test = np.argmax(y_test).reshape(-1, 1)
y_predict = np.argmax(y_predict).reshape(-1, 1)

acc = accuracy_score(y_test, y_predict)

y_submit = model.predict(x_test1)



print('로스 : ', loss[0])
print('정확도 : ', loss[1])
print('시간 :', round(end_time - start_time, 3), '초')


#5. 파일 출력
sampleSubmission['label'] = y_submit
# print(sampleSubmission)

sampleSubmission.to_csv(path+'samplesubmission_0802_1727.csv') #to_csv는 이 데이터를 ~파일을 만들어서 거기에 넣어줄거임

