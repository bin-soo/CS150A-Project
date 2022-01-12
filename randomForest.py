from sklearn.model_selection import KFold,GroupKFold
from dataSeperation import sample3,trainx3,trainy3,trainxs,trainys
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
def rmse(y1,y2):
    return np.sqrt(((y1 - y2) ** 2).mean())
trainx = trainx3
trainy = trainy3
rf = RandomForestClassifier(n_estimators=300, criterion='gini', max_depth=None, min_samples_split=.01, n_jobs=4,
                            random_state=42)
rmseList = []
for kfold, (train, val) in enumerate(KFold(n_splits=5,shuffle=True,random_state=42)\
                                              .split(trainx, trainy)):

    X_train, X_val = trainx.iloc[train], trainx.iloc[val]
    y_train, y_val = trainy.iloc[train], trainy.iloc[val]
    rf.fit(X_train,y_train)
    y_pred = rf.predict_proba(X_val)
    pred = y_pred[:,1]
    print(pred)
    print(pred.shape)
    rmseList.append(rmse(pred,y_val))
print('rmse of val: ',rmseList,'\n mean: ',np.mean(rmseList))

'''
n = 300:
rmse of val:  [0.3581938070589848, 0.3561131152779164, 0.35921866359291854, 0.35509526562366606, 0.3655570216884372]
mean:  0.3588355746483846

'''