import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings(action="once")
import datetime as dt
import os
import sys
import re
sys.path.append("../")

from sklearn.preprocessing import MultiLabelBinarizer
from category_encoders import TargetEncoder

from sklearn.base import BaseEstimator, TransformerMixin 

from joblib import Parallel, delayed
from tqdm import tqdm

class AppFeatureEngineering(BaseEstimator, TransformerMixin):
    """This class is used to execute general ZRP feature engineering.
    
    Parameters
    ----------
    key: str 
        Key to set as index
    first_name: str
        Name of first name column
    middle_name: str
        Name of middle name column
    last_name: str
        Name of last name/surname column
    geo_key: str
        Name of Census GEOID column
    race: str
        Name of race column
    """ 
    
    def __init__(self, key = "ZEST_KEY", geo_key = "GEOID", first_name ="first_name" , middle_name ="middle_name", last_name ="last_name", race = "race"):
        self.key = key
        self.geo_key = geo_key
        self.first_name = first_name
        self.middle_name = middle_name
        self.last_name = last_name
        self.race = race
        
        self.label_encoded_columns = [self.first_name, self.last_name, self.middle_name] # label encode via target
        self.keys = [self.key, self.geo_key]

    def _process_target(self, y): 

        y_unique = y.unique()
        y_unique.sort()
        self.n_classes = len(y_unique)
        
        possible_race_classes = ["AAPI", "AIAN",  "BLACK", "HISPANIC", "WHITE"]
        
        # handle multi-labeled output
        self.mlb = MultiLabelBinarizer(classes = y_unique)
        self.mlb_columns = list(set(possible_race_classes) & set(y_unique))
        print("PRE MLB", y.shape)
        self.mlb.fit(y.values.reshape(-1,1))
        y_ohe = pd.DataFrame(self.mlb.transform(y.values.reshape(-1,1)), columns=self.mlb_columns)
        print("MLB", len(y_ohe))

        self.le = {}
        for i in range(self.n_classes):
            self.le[i] = TargetEncoder()

        return y_ohe
    
    def fit(self, X, y):
        print("App FE", X.shape, y.shape)

        targets = X[[self.key]].merge(y.reset_index(drop=False), on=self.key, how="left")
        y = targets.set_index(self.key)[self.race]
        X = X.reset_index(drop=True)
        print("App FE (post targets)", X.shape, y.shape)
#         y = y.reset_index(drop=True) 

        self.data_columns = list(X.columns)
        self.acs_columns = list(set(self.data_columns) - set(self.label_encoded_columns) - set(self.keys))
        
        y_ohe = self._process_target(y)
        print("App FE (post ohe)", X.shape, y.shape)

        # fit label encoded columns
        for i in range(self.n_classes):
            self.le[i].fit(X[self.label_encoded_columns], y_ohe.iloc[:,i])
        print("App FE (post label encoding)", X.shape, y.shape)
        
        return self
    
    def transform(self, X):


        X = X.reset_index(drop=False)
        print("App FE (in transform)", X.shape)
        data_fe = pd.concat([self.le[i].transform(X[self.label_encoded_columns]) for i in range(self.n_classes)],
                         axis=1, sort=False
                        )
        print("App FE (in transform post data_fe 1)", X.shape)

        data_fe = pd.concat([data_fe,
                          X[self.keys],
                             X[self.acs_columns]
                         ], axis=1, sort=False)
        print("App FE (in transform post data_fe 2)", X.shape)
        label_encoded_colname = []
        for label in self.mlb_columns:
            for col in self.label_encoded_columns:
                label_encoded_colname.append(label + "_" + col)

        data_fe.columns = label_encoded_colname +  self.keys + self.acs_columns
        data_fe[label_encoded_colname] = data_fe[label_encoded_colname].astype(float)

        
        print("App FE (end transform)", X.shape)
        return data_fe
    



      
    
    
class NameAggregation(BaseEstimator, TransformerMixin):
    """
    This class aggregates across expected name columns to impose one-to-one data
    
    Parameters
    ----------
    key: str 
        Key to set as index. If not provided, a key will be generated.
    n_jobs: int (default 1)
        Number of jobs in parallel
    """
    def __init__(self, key, n_jobs):
        self.key = key
        self.n_jobs =  n_jobs
        self.keys = [self.key]

    def fit(self, X, y):
        all_mto_feats = ['AAPI_first_name',
                         'AAPI_last_name',
                         'AAPI_middle_name',
                         'AIAN_first_name',
                         'AIAN_last_name',
                         'AIAN_middle_name',
                         'BLACK_first_name',
                         'BLACK_last_name',
                         'BLACK_middle_name',
                         'HISPANIC_first_name',
                         'HISPANIC_last_name',
                         'HISPANIC_middle_name',
                         'MULTIRACIAL_first_name',
                         'MULTIRACIAL_last_name',
                         'MULTIRACIAL_middle_name',
                         'OTHER_first_name',
                         'OTHER_last_name',
                         'OTHER_middle_name',
                         'WHITE_first_name',
                         'WHITE_last_name',
                         'WHITE_middle_name']
        self.mto_feats = list(set(X.columns).intersection(set(all_mto_feats)))
        self.drop_cols = self.mto_feats 
        return self
        
    def agg_col(self, X):
        da_agg = X.groupby(self.key).agg("mean")
        return(da_agg)
        
    def transform(self, X):
        df = X.copy()
        df = df.filter(self.mto_feats + [self.key])
        
        chunks = [df[x:x+50] for x in range(0, len(df), 50)]
        results = Parallel(n_jobs=90, verbose=1, prefer='threads')(delayed(self.agg_col)(chunk) for chunk in tqdm(chunks))

        aggd_data = pd.concat(results)  

        aggd_columns= list(aggd_data.columns)
        aggd_data[aggd_columns] = aggd_data[aggd_columns].apply(lambda x: x.round(5))
        
        X = X.drop(self.drop_cols, axis=1).drop_duplicates().set_index(self.key)
        aggd_data.sort_index(inplace=True)
        X.sort_index(inplace=True)
        
        data_out = pd.concat([X, aggd_data], axis=1)        
        
        return(data_out)
    