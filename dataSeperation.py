import pandas as pd
import numpy as np
from sklearn.model_selection import KFold,GroupKFold

traindata = pd.read_csv('D:/CS150EVI/CS150A-Project/data/target_clear_train.csv')
sample1 = traindata.sample(n=30000,random_state=42) #FCNN
trainx1 = sample1.iloc[:,1:12]
trainy1 = sample1.iloc[:,12]
traindata = traindata.append(sample1).drop_duplicates(keep=False)
sample2 = traindata.sample(n=30000,random_state=42) #XGBoost
trainx2 = sample2.iloc[:,1:12]
trainy2 = sample2.iloc[:,12]
traindata = traindata.append(sample2).drop_duplicates(keep=False)
sample3 = traindata.sample(n=30000,random_state=42) #randomforest
trainx3 = sample3.iloc[:,1:12]
trainy3 = sample3.iloc[:,12]
traindata = traindata.append(sample3).drop_duplicates(keep=False)
sample4 = traindata.sample(n=30000,random_state=42) #lightGBM
trainx4 = sample4.iloc[:,1:12]
trainy4 = sample4.iloc[:,12]
traindata = traindata.append(sample4).drop_duplicates(keep=False)
sample5 = traindata.sample(n=30000,random_state=42) #autogluon
trainx5 = sample5.iloc[:,1:12]
trainy5 = sample5.iloc[:,12]
trainfinal = traindata.append(sample5).drop_duplicates(keep=False)  #voting
trainx6 = trainfinal.iloc[:,1:12]
trainy6 = trainfinal.iloc[:,12]

samples = pd.concat([sample1,sample2,sample3,sample4,sample5])
trainxs = samples.iloc[:,1:12]
trainys = samples.iloc[:,12]

finaldata = pd.read_csv('D:/CS150EVI/CS150A-Project/data/target_clear_test.csv')   #what we actually need to predict, all test data.
finalx = finaldata.iloc[:,1:12]

testdata = pd.read_csv('D:/CS150EVI/CS150A-Project/data/target_clear_test_label.csv')
testx = testdata.iloc[:,1:12]
testy = testdata.iloc[:,12]
