from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder

import pandas as pd
import numpy as np


traindata = pd.read_table('data/train.csv')
#traindata = traindata.sample(frac=0.1,random_state=42)
trainy = traindata.iloc[:,13]
testdata = pd.read_table('data/test.csv')
testy = testdata.iloc[:,13]
alldata = pd.concat([traindata,testdata])
ally = pd.concat([trainy,testy])
print(traindata.shape)
print(alldata.shape)
print(alldata.count())
alldata.to_csv('data/alldata.csv')

enc =TargetEncoder(cols=['Anon Student Id','Problem Name','Problem Hierarchy','Step Name','KC(Default)'])
to = enc.fit_transform(alldata, ally)
to.to_csv('data/target_all.csv')
totrain = to.iloc[:232744]
totest = to.iloc[232744:233884]
totrain.to_csv('data/target_train.csv')
totest.to_csv('data/target_test.csv')


# traindata = alldata
# encoder = LabelEncoder()
# ID = encoder.fit_transform(traindata['Anon Student Id'].values)
# ID = np.array([ID]).T
# PName = encoder.fit_transform(traindata['Problem Name'].values)
# PName = np.array([PName]).T
# PHierarchy = encoder.fit_transform(traindata['Problem Hierarchy'].values)
# PHierarchy = np.array([PHierarchy]).T
# SName = encoder.fit_transform(traindata['Step Name'].values)
# SName = np.array([SName]).T
# KC = encoder.fit_transform(traindata['KC(Default)'].values)
# KC = np.array([KC]).T
#
# enc = OneHotEncoder()
# ID=enc.fit_transform(ID)
# ID=ID.toarray()
# PName=enc.fit_transform(PName)
# PName=PName.toarray()
# PHierarchy=enc.fit_transform(PHierarchy)
# PHierarchy=PHierarchy.toarray()
# SName=enc.fit_transform(SName)
# SName=SName.toarray()
# KC=enc.fit_transform(KC)
# KC=KC.toarray()


