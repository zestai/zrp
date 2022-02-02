print('\n---\nloading packages')
import os
import re
import sys
import json
import numpy as np
import pandas as pd
from os.path import expanduser
import joblib

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from feature_engine.imputation import MeanMedianImputer
from feature_engine.selection import SmartCorrelatedSelection, DropFeatures

home = expanduser("~")
base_model = 'base_zrp'
model_version  = 'exp_010'
level =  'census_tract'
out_data_path = f"/d/shared/zrp/model_artifacts/experiment/{model_version}/{level}/data/"
src_path = '{}zest-race-predictor/playground/kam/zrp/modeling/models/census_tract/'.format(home)
sys.path.append(src_path)

from src.app_preprocessor import HandleCompoundNames
from src.acs_scaler import CustomRatios
from src.app_fe import AppFeatureEngineering, NameAggregation
from src.set_key import SetKey


# Set default parameters
race = 'race'
first_name = "first_name"
middle_name = "middle_name"
last_name="last_name"
key = "ZEST_KEY"
geo_key = 'GEOID'
n_jobs = -1

def load_json(path):
    with open(path, "r") as infile:
        data = json.load(infile)
    return data

cwd = os.getcwd()
feature_list = load_json(os.path.join(cwd, 'feature_list.json'))
print("Number of initial features", len(feature_list))

sample_path = f"/d/shared/zrp/shared_data/processed/data/sampling/zrp/{level}/"
## Load Data
print('\n---\nloading data')

test_files = ["X_test_FL.feather",
                "X_test_GA.feather",
                "X_test_NC.feather"]
test_targ_files = ["y_test_FL.feather",
                    "y_test_GA.feather",
                    "y_test_NC.feather"]
train_files = ["X_train_FL.feather",
               "X_train_GA.feather",
               "X_train_NC.feather"]
train_targ_files = ["y_train_FL.feather",
                    "y_train_GA.feather",
                    "y_train_NC.feather"]
valid_files = ["X_valid_FL.feather",
                "X_valid_GA.feather",
                "X_valid_NC.feather"]
valid_targ_files = ["y_valid_FL.feather",
                    "y_valid_GA.feather",
                    "y_valid_NC.feather"]

test_dfs = []
for file in test_files:
    tmp = pd.read_feather(os.path.join(sample_path, file))
    tmp = tmp.filter(feature_list)
    test_dfs.append(tmp)
X_test = pd.concat(test_dfs)    

test_targ_dfs = []
for file in test_targ_files:
    tmp = pd.read_feather(os.path.join(sample_path, file))
    test_targ_dfs.append(tmp) 
y_test = pd.concat(test_targ_dfs)    


train_dfs = []
for file in train_files:
    tmp = pd.read_feather(os.path.join(sample_path, file))
    tmp = tmp.filter(feature_list)
    train_dfs.append(tmp)
X_train = pd.concat(train_dfs)    

train_targ_dfs = []
for file in train_targ_files:
    tmp = pd.read_feather(os.path.join(sample_path, file))
    train_targ_dfs.append(tmp)    
y_train = pd.concat(train_targ_dfs)    

        
valid_dfs = []
for file in valid_files:
    tmp = pd.read_feather(os.path.join(sample_path, file))
    tmp = tmp.filter(feature_list)
    valid_dfs.append(tmp)
X_valid = pd.concat(valid_dfs)    

valid_targ_dfs = []
for file in valid_targ_files:
    tmp = pd.read_feather(os.path.join(sample_path, file))
    valid_targ_dfs.append(tmp)    
y_valid = pd.concat(valid_targ_dfs)  

print("")
print("SHAPE:", X_train.shape, y_train.shape)
print("KEYS:", X_train[key].nunique(), y_train[key].nunique())
print("")

X_train_keys = list(X_train[key].unique())
y_train_keys = list(y_train[key].unique())
train_keys = list(set(X_train_keys).intersection(set(y_train_keys)))

X_valid_keys = list(X_valid[key].unique())
y_valid_keys = list(y_valid[key].unique())
valid_keys = list(set(X_valid_keys).intersection(set(y_valid_keys)))

X_test_keys = list(X_test[key].unique())
y_test_keys = list(y_test[key].unique())
test_keys = list(set(X_test_keys).intersection(set(y_test_keys)))

X_train = X_train[X_train[key].isin(train_keys)] 
X_valid = X_valid[X_valid[key].isin(valid_keys)] 
X_test = X_test[X_test[key].isin(test_keys)] 
y_train = y_train[y_train[key].isin(train_keys)] 
y_valid = y_valid[y_valid[key].isin(valid_keys)] 
y_test = y_test[y_test[key].isin(test_keys)] 

print("")
print("SHAPE:", X_train.shape, y_train.shape)
print("KEYS:", X_train[key].nunique(), y_train[key].nunique())
print("")

verify_train = X_train.shape[0] == y_train.shape[0]
verify_valid = X_valid.shape[0] == y_valid.shape[0]
verify_test = X_test.shape[0] == y_test.shape[0]

if not verify_train:
    X_train = X_train.drop_duplicates('ZEST_KEY')
    y_train = y_train.drop_duplicates('ZEST_KEY')

if not verify_valid:
    X_valid = X_valid.drop_duplicates('ZEST_KEY')
    y_valid = y_valid.drop_duplicates('ZEST_KEY')
    
if not verify_test:
    X_test = X_test.drop_duplicates('ZEST_KEY')
    y_test = y_test.drop_duplicates('ZEST_KEY')
    
print("Shapes are the same:", verify_train,
verify_valid,
verify_test)
    
print("")
sample_weights = y_train[[key, 'sample_weight']].copy() 


print("   How much data do we expect to train off of:", X_train.shape )

#### Convert keys to strings
print('\n---\nformatting data')
X_train[[geo_key, key]] = X_train[[geo_key, key]].astype(str)
y_train[[geo_key, key]] = y_train[[geo_key, key]].astype(str)
sample_weights[key] = sample_weights[key].astype(str)


#### Set Index
print("   set indices")
X_train.set_index(key, inplace=True)
y_train.set_index(key, inplace=True)
sample_weights.set_index(key, inplace=True)
X_train.sort_index(inplace=True)
y_train.sort_index(inplace=True)
sample_weights.sort_index(inplace=True)

X_valid.set_index(key, inplace=True)
y_valid.set_index(key, inplace=True)
X_valid.sort_index(inplace=True)
y_valid.sort_index(inplace=True)

X_test.set_index(key, inplace=True)
y_test.set_index(key, inplace=True)
X_test.sort_index(inplace=True)
y_test.sort_index(inplace=True)
print("")
print("SHAPE:", X_train.shape, y_train.shape)
print("")

cols = list(set(X_train.columns) - set([key, geo_key, 'GEOID_BG',  'GEOID_CT',
                                        'GEOID_ZIP', "first_name", "middle_name", 
                                        "last_name", 'ZEST_KEY_COL']))

print('   train to numeric')
X_train[cols] = X_train[cols].apply(pd.to_numeric, errors='coerce')
print('   valid to numeric')

X_valid[cols] = X_valid[cols].apply(pd.to_numeric, errors='coerce')
print('   test to numeric')

X_test[cols] = X_test[cols].apply(pd.to_numeric, errors='coerce')

print("")

X_train.reset_index(drop=False).to_feather(os.path.join(out_data_path, f"train_raw_data.feather"))
X_valid.reset_index(drop=False).to_feather(os.path.join(out_data_path, f"valid_raw_data.feather"))
X_test.reset_index(drop=False).to_feather(os.path.join(out_data_path, f"test_raw_data.feather"))

print("")
print("SHAPE:", X_train.shape, y_train.shape)
print("")

### Build Pipeline
print("...building pipeline")
import time
start_time = time.time()

pipe = Pipeline(
    [("Drop Features", DropFeatures(features_to_drop = ['GEOID_BG', 'GEOID_CT', 'GEOID_ZIP', 'ZEST_KEY_COL'])),
     ("Compound Name FE", HandleCompoundNames(last_name = last_name, first_name = first_name, middle_name = middle_name)),
     ("App FE", AppFeatureEngineering( key = key, geo_key = geo_key, first_name = first_name , middle_name = middle_name, last_name = last_name, race = race)),
     ("ACS FE", CustomRatios()), 
     ("Name Aggregation", NameAggregation(key = key, n_jobs = n_jobs)),
     ("Drop Features (2)", DropFeatures(features_to_drop = ['GEOID'])),
     ("Impute", MeanMedianImputer(imputation_method = "mean", variables=None)), # numeric only variable=None implies imputer will impute all numeric variables may need to fitler depending on if non-numeric features in the model
     ("Correlated Feature Selection",  SmartCorrelatedSelection(method = 'pearson', threshold=.95))],
    verbose=True
)


start_fit_time = time.time()
print(start_fit_time)
#### Fit the Pipeline
print('\n---\nfitting pipeline')
pipe.fit(X_train, y_train[race]) 

import pickle 
import joblib
pickle.dump(pipe,open( os.path.join(out_data_path, 'pipe.pkl'), 'wb'))
joblib.dump(pipe, os.path.join(out_data_path, 'pipe.joblib'))

end_fit_time = time.time()
print("Total Time:", end_fit_time - start_fit_time, "\n",
     "Total Time Adjusted:", (end_fit_time - start_fit_time)/60, "\n",
    "End Time:", end_fit_time)

#### Transform
##### This step creates the feature engineering data
print('\n---\ntransforming fe data')
start_transform_time = time.time()
print(start_transform_time)
print("")
X_train_fe = pipe.transform(X = X_train)


end_transform_time = time.time()
print("Total Time:", end_transform_time - start_transform_time, "\n",
     "Total Time Adjusted:", (end_transform_time - start_transform_time)/60, "\n",
    "End Time:", end_transform_time)

### Build the Model

##### specify model parameters
print('\n---\nbuilding model')
start_model_time = time.time()
print(start_model_time)


opt_params = {'gamma': 5,
              'learning_rate': 0.01, 
              'max_depth':3,
              'min_child_weight': 500,
              'n_estimators': 2000,
              'subsample': 0.20}

##### Initialize the model

CLF = XGBClassifier(objective='multi:softprob',
                    num_class=len(y_train[race].unique()),
                    **opt_params)


##### Fit
print('\n---\nfitting model')
CLF.fit(
    X_train_fe,  y_train[race],
        sample_weight = sample_weights.sample_weight
)

end_model_time = time.time()
print("Total Time:", end_model_time - start_model_time, "\n",
     "Total Time Adjusted:", (end_model_time - start_model_time)/60, "\n",
    "End Time:", end_model_time)

print("True Race Distribution: \n", y_train[race].value_counts(dropna=False))
print(X_train_fe.shape, y_train.shape, len(sample_weights))



CLF.save_model(f'/d/shared/zrp/model_artifacts/experiment/{model_version}/{level}/data/xgb_model.json')
#save model
joblib.dump(CLF, f"/d/shared/zrp/model_artifacts/experiment/{model_version}/{level}/data/model.joblib") 

X_valid_fe = pipe.transform(X = X_valid)
X_test_fe = pipe.transform(X = X_test)

X_train_fe.reset_index(drop=False).to_feather(os.path.join(out_data_path, f"train_fe_data.feather"))
X_valid_fe.reset_index(drop=False).to_feather(os.path.join(out_data_path, f"valid_fe_data.feather"))
X_test_fe.reset_index(drop=False).to_feather(os.path.join(out_data_path, f"test_fe_data.feather"))

##### Return Race Probabilities
print('\n---\nrace predictions')
y_hat_train = pd.DataFrame({'race' :CLF.predict(X_train_fe)}, index=y_train.index)
y_hat_valid = pd.DataFrame({'race' :CLF.predict(X_valid_fe)}, index=y_valid.index)
y_hat_test = pd.DataFrame({'race' :CLF.predict(X_test_fe)}, index=y_test.index)

y_hat_train.reset_index(drop=False).to_feather(os.path.join(out_data_path, f"train_proxies.feather"))
y_hat_valid.reset_index(drop=False).to_feather(os.path.join(out_data_path, f"valid_proxies.feather"))
y_hat_test.reset_index(drop=False).to_feather(os.path.join(out_data_path, f"test_proxies.feather"))

print('\n---\nrace predictions')
y_phat_train = pd.DataFrame(CLF.predict_proba(X_train_fe), index=y_train.index)
y_phat_valid = pd.DataFrame(CLF.predict_proba(X_valid_fe), index=y_valid.index)
y_phat_test = pd.DataFrame(CLF.predict_proba(X_test_fe), index=y_test.index)


y_phat_train.reset_index(drop=False).to_feather(os.path.join(out_data_path, f"train_proxy_probs.feather"))
y_phat_valid.reset_index(drop=False).to_feather(os.path.join(out_data_path, f"valid_proxy_probs.feather"))
y_phat_test.reset_index(drop=False).to_feather(os.path.join(out_data_path, f"test_proxy_probs.feather"))

end_time = time.time()
print("Total Time:", end_time - start_time, "\n",
     "Total Time Adjusted:", (end_time - start_time)/60, "\n",
    "End Time:", end_time)



