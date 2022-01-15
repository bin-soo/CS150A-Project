from sklearn.model_selection import KFold,GroupKFold
from dataSeperation import sample3,trainx5,trainy5,trainxs,trainys
from sklearn.svm import LinearSVC,SVC
import numpy as np
import pandas as pd
import joblib
def rmse(y1,y2):
    return np.sqrt(((y1 - y2) ** 2).mean())
trainx = trainxs
trainy = trainys

svc = SVC(C=0.9,random_state=42,probability=True)
rmseList = []
print('--------------SVC-----------------')
for kfold, (train, val) in enumerate(KFold(n_splits=5,shuffle=True,random_state=42)\
                                              .split(trainx, trainy)):

    X_train, X_val = trainx.iloc[train], trainx.iloc[val]
    y_train, y_val = trainy.iloc[train], trainy.iloc[val]
    svc.fit(X_train,y_train)
    y_pred = svc.predict_proba(X_val)
    pred = y_pred[:,1]
    # print(pred)
    # print(pred.shape)
    rmseList.append(rmse(pred,y_val))

print('rmse of val: ',rmseList,'\nmean: ',np.mean(rmseList))
meansvc = np.mean(rmseList)
joblib.dump(filename='svc1.model',value=svc)
joblib.dump(filename='meansvc1.model',value=meansvc)

'''
C= 0.9:
rmse of val:  [0.4002390056827559, 0.39866664815721725, 0.40782578753152293, 0.41062120920115386, 0.4104773117225855] 
mean:  0.405565992459047

on sample s:

'''
