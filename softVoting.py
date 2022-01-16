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
svc = joblib.load('svc.model')
meansvc = joblib.load('meansvc.model')
xgb = joblib.load('xgb1.model')
meanxgb = joblib.load('meanxgb1.model')
rf = joblib.load('rf1.model')
meanrf = joblib.load('meanrf1.model')
lgbm = joblib.load('gbm1.model')
meangbm = joblib.load('meangbm1.model')
# fcnn = models.load_model('fcnn')
# meannn = joblib.load('meannn.model')
rmseList = []

# experts = [('svc',svc),('xgb',xgb),('rf',rf),('fcnn',fcnn),('lgbm',lgbm)]
# weights = [1/meansvc, 1/meanxgb, 1/meanrf, 1/meannn, 1/meangbm]
experts = [('xgb',xgb),('rf',rf),('lgbm',lgbm)]
weights = [ 1/meanxgb, 1.6/meanrf, 2/meangbm]
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
joblib.dump(filename='soft1.model',value=soft)
joblib.dump(filename='meansoft1.model',value=meanvoting)

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

bagging, weights = [ 1.1/meanxgb, 1.6/meanrf, 2.1/meangbm]:
rmse of val:  [0.3505979341090136, 0.3474659067174202, 0.3509152247289158, 0.3553741188766689, 0.3545079173773185] 
mean:  0.3517722203618674

bagging, weights = [ 1.2/meanxgb, 1.6/meanrf, 2.2/meangbm]:
rmse of val:  [0.35050006915759757, 0.34736900157088463, 0.3508192418127122, 0.35527964503014914, 0.35441468392286696] 
mean:  0.35167652829884216

-------------------------------------

stacking, weights = [ 1/meanxgb, 1.5/meanrf, 2.5/meangbm]:  
rmse of val:  [0.3475475801401125, 0.34480149116232256, 0.3479717383157818, 0.35260771937848534, 0.35216834324609786] 
mean:  0.34901937444856

stacking, weights = [ 1/meanxgb, 1.6/meanrf, 2/meangbm]:
rmse of val:  [0.34759551983083553, 0.3448709656201101, 0.3480598436043383, 0.3526715194345183, 0.35226777525633185] 
mean:  0.349093124749226

stacking, weights = [ 1/meanxgb, 1.8/meanrf, 2.5/meangbm]:
rmse of val:  [0.3479029609447226, 0.34512392154532046, 0.34832362922961074, 0.3529373217081463, 0.35246955796808854] 
mean:  0.3493514782791777

stacking, weights = [ 1.2/meanxgb, 1.5/meanrf, 2.5/meangbm]: @@@@@
rmse of val:  [0.3471281012576986, 0.3444475069907236, 0.3475929551674653, 0.35223742066553826, 0.3518689586242636] 
mean:  0.34865498854113786

stacking, weights = [ 1/meanxgb, 1.5/meanrf, 2.2/meangbm]:
rmse of val:  [0.34749177896810796, 0.3447658733414905, 0.3479403202370345, 0.352566809671724, 0.3521549641599171] 
mean:  0.34898394927565485

stacking, weights = [ 1.1/meanxgb, 1.6/meanrf, 2.1/meangbm]:
rmse of val:  [0.3473761820379138, 0.34467914025023283, 0.34785272181768223, 0.35247237128644915, 0.35209731418875645] 
mean:  0.3488955459162069

'''
