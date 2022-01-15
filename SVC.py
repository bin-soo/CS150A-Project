from sklearn.model_selection import KFold,GroupKFold
from dataSeperation import sample3,trainx5,trainy5,trainxs,trainys
from sklearn.svm import LinearSVC,SVC
import numpy as np
import pandas as pd
def rmse(y1,y2):
    return np.sqrt(((y1 - y2) ** 2).mean())
trainx = trainx5
trainy = trainy5

rf = SVC(C=1.3,random_state=42,probability=True)
rmseList = []
for kfold, (train, val) in enumerate(KFold(n_splits=4,shuffle=True,random_state=42)\
                                              .split(trainx, trainy)):

    X_train, X_val = trainx.iloc[train], trainx.iloc[val]
    y_train, y_val = trainy.iloc[train], trainy.iloc[val]
    rf.fit(X_train,y_train)
    y_pred = rf.predict_proba(X_val)
    pred = y_pred[:,1]
    print(pred)
    print(pred.shape)
    rmseList.append(rmse(pred,y_val))
print('rmse of val: ',rmseList,'\nmean: ',np.mean(rmseList))

'''
C= 0.9:
rmse of val:  [0.4002390056827559, 0.39866664815721725, 0.40782578753152293, 0.41062120920115386, 0.4104773117225855] 
mean:  0.405565992459047

C = 0.9
rmse of val:  [0.4002523554019535, 0.39869706832686136, 0.40783658901547903, 0.4106181325867557, 0.4105372324381642]
mean:  0.4055882755538427

C = 1.2 ORIGIN N = 5
rmse of val:  [0.40009337497638675, 0.39869007028867354, 0.4078004768408474, 0.41038309106699744, 0.4104705295531215]
mean:  0.4054875085452053

c = 1.2 N_SPLITS = 6
rmse of val:  [0.4005077994491544, 0.3958040953120146, 0.4070609776595962, 0.4036532383800154, 0.4133046936223944, 0.4125531522432226]
mean:  0.40548065944439954

1.4 5
rmse of val:  [0.4000890894234751, 0.398667660229254, 0.40777485529664104, 0.4103573638885193, 0.41043492594052833]
mean:  0.4054647789556835

1.7 5
rmse of val:  [0.4001274559926941, 0.3987764748188913, 0.4077612392953167, 0.4103131684111502, 0.41039130180296396]
mean:  0.4054739280642033

1.5 5
rmse of val:  [0.4000981952713851, 0.3987097896731527, 0.40774984792479596, 0.41038891442245423, 0.41040062884557]
mean:  0.40546947522747157

1.3 5
rmse of val:  [0.4000851985315604, 0.3986833254791853, 0.40777966210257743, 0.41031393981191727, 0.41046815964981825]
mean:  0.40546605711501177

1.3 4
rmse of val:  [0.40052742674697955, 0.40150516285618293, 0.41057364133364144, 0.40934671653106824]
mean:  0.40548823686696805
'''