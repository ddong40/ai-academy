import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

path = 'C:/Users/ddong40/ai_2/_data/dacon/따릉이/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0) 
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 
sampleSubmission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv.columns)
print(train_csv.info)

print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isna().sum())

test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop