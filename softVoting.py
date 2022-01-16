from dataSeperation import trainx6,trainy6
# from lightGBM import lgbm, meangbm
# from FCNN import fcnn, meannn
# from randomForest import rf, meanrf
# from SVC import svc, meansvc
# from XGB import xgb, meanxgb
from keras import models
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold
def rmse(y1,y2):
    return np.sqrt(((y1 - y2) ** 2).mean())
svc = joblib.load('D:/CS150EVI/CS150A-Project/models/svc.model')
meansvc = joblib.load('D:/CS150EVI/CS150A-Project/models/meansvc.model')
xgb = joblib.load('D:/CS150EVI/CS150A-Project/models/xgb.model')
meanxgb = joblib.load('D:/CS150EVI/CS150A-Project/models/meanxgb.model')
rf = joblib.load('D:/CS150EVI/CS150A-Project/models/rf.model')
meanrf = joblib.load('D:/CS150EVI/CS150A-Project/models/meanrf.model')
lgbm = joblib.load('D:/CS150EVI/CS150A-Project/models/gbm.model')
meangbm = joblib.load('D:/CS150EVI/CS150A-Project/models/meangbm.model')
# fcnn = models.load_model('fcnn')
# meannn = joblib.load('meannn.model')
rmseList = []

# experts = [('svc',svc),('xgb',xgb),('rf',rf),('fcnn',fcnn),('lgbm',lgbm)]
# weights = [1/meansvc, 1/meanxgb, 1/meanrf, 1/meannn, 1/meangbm]
experts = [('xgb',xgb),('rf',rf),('lgbm',lgbm)]
weights = [ 1/meanxgb, 1.5/meanrf, 2.5/meangbm]
soft = VotingClassifier(estimators=experts,voting='soft',weights=weights)
print('--------------soft voting-----------------')

for kfold, (train, val) in enumerate(KFold(n_splits=5,shuffle=True,random_state=42)\
                                              .split(trainx6, trainy6)):
    print('kfold start.')
    X_train, X_val = trainx6.iloc[train], trainx6.iloc[val]
    y_train, y_val = trainy6.iloc[train], trainy6.iloc[val]
    soft.fit(X_train,y_train)
    y_pred = soft.predict_proba(X_val)
    pred = y_pred[:,1]
    rmseList.append(rmse(pred,y_val))

print('rmse of val: ',rmseList,'\nmean: ',np.mean(rmseList))
meanvoting = np.mean(rmseList)
joblib.dump(filename='D:/CS150EVI/CS150A-Project/models/soft1.model',value=soft)
joblib.dump(filename='D:/CS150EVI/CS150A-Project/models/meansoft1.model',value=meanvoting)

'''
bagging, equal weights of 4:
rmse of val:  [0.3554231972073515, 0.35212108740733583, 0.35671526196272807, 0.3605985694074069, 0.35948518178309175] 
mean:  0.3568686595535828

bagging, weights = [ 1/meanxgb, 1.5/meanrf, 2/meangbm]:
rmse of val:  [0.3506116092964202, 0.3474794986107349, 0.35092505580635774, 0.3553861477848396, 0.35452059090338645] 
mean:  0.3517845804803478

bagging, weights = [ 1/meanxgb, 1.5/meanrf, 2.5/meangbm]:
rmse of val:  [0.3504082216865059, 0.34727920880477003, 0.3506895911819205, 0.3551787714621006, 0.354323565059915] 
mean:  0.3515758716390424
'''