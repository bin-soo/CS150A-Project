# CS150A-Project  

## Data Preprocessing with PySpark
### feature engineering
1. extract similarity or feature of KC  
2. Add features: CFAR/Duration for class features.  
### dataset seperation 
k-fold cross validation for each expert  
## Classification
### Bagging - voting of L experts (classifiers)
with/without dropout in val for bagging  
Base expert:
  ` Autogluon
  ` FCNN
  ` XGBoost
  ` RandomForest
  ` LightGBM
