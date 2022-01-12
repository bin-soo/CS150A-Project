# CS150A-Project  

## Data Preprocessing with PySpark
### Data Cleaning
1. missing value:
  Duration(mean:(student+problemname+step)/3),KC(0),Oppotunity(0).
  
### feature engineering
1. one-hot encoding for text features.  
2. extract similarity or feature of KC.  
3. Add features: ID+PH+PN+PV+SN+KC+(OP每单元求和）+8x(student CFAR/problem CFAR/step CFAR/PH CFAR/Duration)+(~KC聚类) for text features.  
### dataset seperation
k-fold cross validation for each expert  
## Classification
### Bagging - soft voting of L experts (classifiers) with base acc as weight
with/without dropout in val for bagging  
Base expert:
  - Autogluon
  - FCNN
  - XGBoost
  - RandomForest
  - LightGBM
  - kernel SVM
