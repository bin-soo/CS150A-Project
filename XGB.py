import pandas as pd
import numpy as np
from sklearn.model_selection import KFold,GroupKFold
import xgboost as xgb
from xgboost import plot_importance
from dataSeperation import sample2,trainx2,trainy2,trainxs,trainys
def rmse(y1,y2):
    return np.sqrt(((y1 - y2) ** 2).mean())
trainx = trainxs
trainy = trainys
xgb = xgb.XGBClassifier(max_depth=7, learning_rate=0.08, n_estimators=200, objective='binary:logistic',
                          random_state=42)
rmseList = []
for kfold, (train, val) in enumerate(KFold(n_splits=5,shuffle=True,random_state=42)\
                                              .split(trainx, trainy)):

    X_train, X_val = trainx.iloc[train], trainx.iloc[val]
    y_train, y_val = trainy.iloc[train], trainy.iloc[val]
    xgb.fit(X_train,y_train)
    y_pred = xgb.predict(X_val)
    rmseList.append(rmse(y_pred,y_val))
print('rmse of val: ',rmseList,'\nmean: ',np.mean(rmseList))

'''
depth = 0.7, lr=0.08, n=200:
rmse of val:  [0.4002390056827559, 0.39866664815721725, 0.40782578753152293, 0.41062120920115386, 0.4104773117225855] 
 mean:  0.405565992459047
'''
