# CS150A-Project  

## Data Preprocessing with PySpark
### feature engineering
1. one-hot encoding for text features.  
2. extract similarity or feature of KC.  
3. Add features: CFAR/Duration for text features.  
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
  - SVM
