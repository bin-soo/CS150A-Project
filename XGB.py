import pandas as pd
import numpy as np
from sklearn.model_selection import KFold,GroupKFold
import xgboost as xgb
from xgboost import plot_importance
from dataSeperation import sample2,trainx2,trainy2,trainxs,trainys
import joblib
def rmse(y1,y2):
    return np.sqrt(((y1 - y2) ** 2).mean())
trainx = trainxs
trainy = trainys
xgb = xgb.XGBClassifier(max_depth=10, learning_rate=0.12, n_estimators=500, objective='binary:logistic',
                          random_state=42)
rmseList = []
print('--------------XGB-----------------')

for kfold, (train, val) in enumerate(KFold(n_splits=5,shuffle=True,random_state=42)\
                                              .split(trainx, trainy)):

    X_train, X_val = trainx.iloc[train], trainx.iloc[val]
    y_train, y_val = trainy.iloc[train], trainy.iloc[val]
    xgb.fit(X_train,y_train)
    y_pred = xgb.predict(X_val)
    rmseList.append(rmse(y_pred,y_val))
print('rmse of val: ',rmseList,'\nmean: ',np.mean(rmseList))
meanxgb = np.mean(rmseList)
joblib.dump(filename='xgb1.model',value=xgb)
joblib.dump(filename='meanxgb1.model',value=meanxgb)
'''
depth = 7, lr=0.08, n=200:
rmse of val:  [0.4002390056827559, 0.39866664815721725, 0.40782578753152293, 0.41062120920115386, 0.4104773117225855] 
 mean:  0.405565992459047

on sample s:
rmse of val:  [0.4166533331199932, 0.41255302689472534, 0.4148493702538308, 0.4124722212868805, 0.4143267631552018] 
mean:  0.41417094294212636

depth = 10, lr=0.12, n=500, on sample s:
rmse of val:  [0.41158231254513356, 0.40591460514086786, 0.407635458058627, 0.4086155487333622, 0.4091861841916627] 
mean:  0.4085868217339307
'''
