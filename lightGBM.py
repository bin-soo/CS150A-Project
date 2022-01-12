from sklearn.model_selection import KFold,GroupKFold
from dataSeperation import sample3,trainx4,trainy4,trainxs,trainys
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import lightgbm as gbm
def rmse(y1,y2):
    return np.sqrt(((y1 - y2) ** 2).mean())
trainx = trainxs
trainy = trainys
lgbm = gbm.LGBMClassifier(boosting_type = 'gbdt', objective='binary', max_depth=5, num_leaves = 45, learning_rate=0.08, n_estimators=300, n_jobs=4,
                         min_child_weight=1, seed=42, subsample=0.85, subsample_freq = 1, boost_from_average = False, reg_lambda = 0.12)
rmseList = []
for kfold, (train, val) in enumerate(KFold(n_splits=5,shuffle=True,random_state=42)\
                                              .split(trainx, trainy)):

    X_train, X_val = trainx.iloc[train], trainx.iloc[val]
    y_train, y_val = trainy.iloc[train], trainy.iloc[val]
    lgbm.fit(X_train,y_train)
    y_pred = lgbm.predict_proba(X_val)
    pred = y_pred[:,1]
    print(pred)
    print(pred.shape)
    rmseList.append(rmse(pred,y_val))
print('rmse of val: ',rmseList,'\nmean: ',np.mean(rmseList))

'''
n = 300, lr = 0.08, depth = 5:
rmse of val:  [0.3559166920013514, 0.35734895683093776, 0.3588461908007403, 0.3575335236367929, 0.3528417851410494] 
mean:  0.35649742968217435

on sample s:
rmse of val:  [0.3497693321701627, 0.3473746597287868, 0.3479416654365751, 0.34837133143416604, 0.3494885287580998] 
mean:  0.34858910350555805
-----
n = 300, lr = 0.1, depth = 5:
rmse of val:  [0.35685672113496325, 0.3587061077808856, 0.3607680583827991, 0.3579740703276941, 0.35410656529477524] 
mean:  0.35768230458422345
'''