print('\n---\nloading packages')
import os
import re
import sys
import json
import numpy as np
import pandas as pd
from os.path import expanduser
import joblib
import pickle 

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from feature_engine.imputation import MeanMedianImputer
from feature_engine.selection import SmartCorrelatedSelection, DropFeatures

home = expanduser("~")
model_version  = 'exp_013'
level =  'census_tract'

out_data_path = f"/d/shared/zrp/model_artifacts/experiment/{model_version}/{level}/data/"
gcwd = os.getcwd()
if 'home' not in gcwd:
    home = '/d/shared'
    
src_path = f'{home}/zrp/zrp/modeling/models/{level}/'
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

#cwd = f"{home}/zrp/zrp/modeling/models/{level}/"
cwd = os.getcwd()
feature_list = load_json(os.path.join(cwd, 'feature_list.json'))
fl_ids = load_json(os.path.join(cwd, 'fl_ids.json'))

sample_path = f"/d/shared/zrp/model_artifacts/experiment/exp_010/{level}/data"

## Load Data
print('\n---\nloading data\n---\n')

X_test = pd.read_feather(os.path.join(sample_path, f"test_raw_data.feather"))
X_train = pd.read_feather(os.path.join(sample_path, f"train_raw_data.feather"))
X_valid = pd.read_feather(os.path.join(sample_path, f"valid_raw_data.feather"))

y_test = pd.read_feather(os.path.join(sample_path, f"test_raw_targets.feather"))
y_train = pd.read_feather(os.path.join(sample_path, f"train_raw_targets.feather"))
y_valid = pd.read_feather(os.path.join(sample_path, f"valid_raw_targets.feather"))


y_train[[geo_key, key]] = y_train[[geo_key, key]].astype(str)
sample_weights = y_train[[key, 'sample_weight']].copy() 

print("")
print("SHAPE:", X_train.shape, y_train.shape)
print("KEYS:", X_train[key].nunique(), y_train[key].nunique())
print("")


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

X_train.reset_index(drop=False).to_feather(os.path.join(out_data_path, f"train_raw_data.feather"))
X_valid.reset_index(drop=False).to_feather(os.path.join(out_data_path, f"valid_raw_data.feather"))
X_test.reset_index(drop=False).to_feather(os.path.join(out_data_path, f"test_raw_data.feather"))

y_train.reset_index(drop=False).to_feather(os.path.join(out_data_path, f"train_raw_targets.feather"))
y_valid.reset_index(drop=False).to_feather(os.path.join(out_data_path, f"valid_raw_targets.feather"))
y_test.reset_index(drop=False).to_feather(os.path.join(out_data_path, f"test_raw_targets.feather"))

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
     ("Impute", MeanMedianImputer(imputation_method = "mean", variables=None)),
     ("Correlated Feature Selection",  SmartCorrelatedSelection(method = 'pearson', threshold=.95))],
    verbose=True
)


start_fit_time = time.time()
print(start_fit_time)
#### Fit the Pipeline
print('\n---\nfitting pipeline')
pipe.fit(X_train, y_train[race]) 


end_fit_time = time.time()
print("Total Time:", end_fit_time - start_fit_time, "\n",
     "Total Time Adjusted:", (end_fit_time - start_fit_time)/60, "\n",
    "End Time:", end_fit_time)

# Save pipeline
pickle.dump(pipe,open( os.path.join(out_data_path, 'pipe.pkl'), 'wb'))
joblib.dump(pipe, os.path.join(out_data_path, 'pipe.joblib'))



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

# Save train fe data
X_train_fe.reset_index(drop=False).to_feather(os.path.join(out_data_path, f"train_fe_data.feather"))



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
              'subsample': 0.60}

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

# Save model
CLF.save_model(f'/d/shared/zrp/model_artifacts/experiment/{model_version}/{level}/data/xgb_model.json')
#save model
joblib.dump(CLF, f"/d/shared/zrp/model_artifacts/experiment/{model_version}/{level}/data/model.joblib") 


# transform & save remaining splits
X_valid_fe = pipe.transform(X = X_valid)
X_test_fe = pipe.transform(X = X_test)

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


print('\n---\nrace predictions (probabilities)')
y_phat_train = pd.DataFrame(CLF.predict_proba(X_train_fe), index=y_train.index)
y_phat_valid = pd.DataFrame(CLF.predict_proba(X_valid_fe), index=y_valid.index)
y_phat_test = pd.DataFrame(CLF.predict_proba(X_test_fe), index=y_test.index)

y_phat_train.columns = ['RACE_' + str(col) for col in y_phat_train.columns]
y_phat_valid.columns = ['RACE_' + str(col) for col in y_phat_valid.columns]
y_phat_test.columns = ['RACE_' + str(col) for col in y_phat_test.columns]

y_phat_train.reset_index(drop=False).to_feather(os.path.join(out_data_path, f"train_proxy_probs.feather"))
y_phat_valid.reset_index(drop=False).to_feather(os.path.join(out_data_path, f"valid_proxy_probs.feather"))
y_phat_test.reset_index(drop=False).to_feather(os.path.join(out_data_path, f"test_proxy_probs.feather"))

end_time = time.time()
print("Total Time:", end_time - start_time, "\n",
     "Total Time Adjusted:", (end_time - start_time)/60, "\n",
    "End Time:", end_time)




