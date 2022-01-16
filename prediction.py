from dataSeperation import finalx, testx, testy, trainx1,trainy1
from sklearn.ensemble import VotingClassifier
import joblib
import numpy as np
import pandas as pd
def rmse(y1,y2):
    return np.sqrt(((y1 - y2) ** 2).mean())
print(testx.count())
print(finalx.count())
soft = joblib.load(filename='soft1.model')

pred_y = soft.predict_proba(testx)
pred = pred_y[:,1]
loss = rmse(pred,testy)
print('rmse on testset in prob: ',loss)
pred2_y = soft.predict(testx)
loss = rmse(pred2_y,testy)
print('rmse on testset: ',loss)

finaly = soft.predict_proba(finalx)
final = finaly[:,1]
pd_final = pd.DataFrame(final)
pd_final.to_csv('best results.csv')

'''
stacking weights = [ 1/meanxgb, 1.6/meanrf, 2/meangbm]:@@@@@
rmse on testset in prob:  0.3454051701334339
rmse on testset:  0.4137283784938751

stacking, weights = [ 1/meanxgb, 1.5/meanrf, 2.2/meangbm]:
rmse on testset in prob:  0.3455555969834573
rmse on testset:  0.4119097834109669

stacking, weights = [ 1.2/meanxgb, 1.5/meanrf, 2.5/meangbm]:
rmse on testset in prob:  0.3457457922532777
rmse on testset:  0.4100831234861637

stacking, weights = [ 1.1/meanxgb, 1.6/meanrf, 2.1/meangbm]:
rmse on testset in prob:  0.345483313885802
rmse on testset:  0.4137283784938751

---------------------------------

bagging, weights = [ 1/meanxgb, 1.5/meanrf, 2.5/meangbm]:
rmse on testset in prob:  0.34605421804291403
rmse on testset:  0.4026936331284146

bagging, weights = [ 1.1/meanxgb, 1.6/meanrf, 2.1/meangbm]:
rmse on testset in prob:  0.34586393252227404
rmse on testset:  0.4045536598075262

bagging, weights = [ 1.2/meanxgb, 1.6/meanrf, 2.2/meangbm]:
rmse on testset in prob:  0.3458977883205284
rmse on testset:  0.4045536598075262

bagging, weights = [ 1/meanxgb, 1.5/meanrf, 2/meangbm]:
rmse on testset in prob:  0.3458766951484441
rmse on testset:  0.4045536598075262
'''

