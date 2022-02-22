from os.path import expanduser, dirname
import pandas as pd
import numpy as np
import os
import re
import sys
import json
import joblib
import pickle

import xgboost
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from feature_engine.imputation import MeanMedianImputer
from feature_engine.selection import SmartCorrelatedSelection, DropFeatures

from zrp.modeling.src.app_preprocessor import HandleCompoundNames
from zrp.modeling.src.acs_scaler import CustomRatios
from zrp.modeling.src.app_fe import AppFeatureEngineering, NameAggregation
from zrp.modeling.src.set_key import SetKey
from zrp.prepare.utils import load_json, load_file, save_feather, make_directory
from zrp.prepare.base import BaseZRP
from zrp.prepare.prepare import ZRP_Prepare

import warnings
warnings.filterwarnings(action='ignore')

curpath = dirname(__file__)


class ZRP_Build_Pipeline(BaseZRP):
    """
    Fits a new ZRP pipeline from user input 
    
    Parameters
    ----------
    file_path: str, optional
        Path indicating where to put artifacts folder its files (pipeline, model, and supporting data), generated during intermediate steps.
    zrp_model_name: str
        Name of zrp_model
    zrp_model_source: str
        Indicates the source of zrp_modeling data to use. There are three optional sources 'bg' (for block_group), 'ct' (for census_tract), and 'zp' (for zip_code). By default 'census_tract' is inferred.
    """

    def __init__(self, file_path=None, zrp_model_name='zrp_0', zrp_model_source='ct', *args, **kwargs):
        super().__init__(file_path=file_path, *args, **kwargs)
        self.zrp_model_name = zrp_model_name
        self.zrp_model_source = zrp_model_source
        self.outputs_path = os.path.join(self.out_path,
                                         "experiments",
                                         self.zrp_model_name,
                                         self.zrp_model_source,
                                         "data")
        self.geo_key = 'GEOID'

    def fit(self, X, y):
        ### Build Pipeline
        print('\n---\nBuilding pipeline')

        self.pipe = Pipeline(
            [("Drop Features", DropFeatures(features_to_drop=['GEOID_BG', 'GEOID_CT', 'GEOID_ZIP', 'ZEST_KEY_COL'])),
             ("Compound Name FE",
              HandleCompoundNames(last_name=self.last_name, first_name=self.first_name, middle_name=self.middle_name)),
             ("App FE", AppFeatureEngineering(key=self.key, geo_key=self.geo_key, first_name=self.first_name,
                                              middle_name=self.middle_name, last_name=self.last_name, race=self.race)),
             ("ACS FE", CustomRatios()),
             ("Name Aggregation", NameAggregation(key=self.key, n_jobs=self.n_jobs)),
             ("Drop Features (2)", DropFeatures(features_to_drop=['GEOID'])),
             ("Impute", MeanMedianImputer(imputation_method="mean", variables=None)),
             ("Correlated Feature Selection", SmartCorrelatedSelection(method='pearson',
                                                                       threshold=.95))],
            verbose=True
        )
        #### Fit the Pipeline
        print('\n---\nFitting pipeline')
        self.pipe.fit(X, y[self.race])

        return self

    def transform(self, X):
        make_directory(self.outputs_path)
        # Save pipeline
        pickle.dump(self.pipe, open(os.path.join(self.outputs_path, 'pipe.pkl'), 'wb'))
        #### Transform
        ##### This step creates the feature engineering data
        print('\n---\nTransforming FE data')
        X_train_fe = self.pipe.transform(X=X)

        # Save train fe data
        print('\n---\nSaving FE data')
        save_feather(X_train_fe, self.outputs_path, f"train_fe_data.feather")
        return (X_train_fe)


class ZRP_Build_Model(BaseZRP):
    """
    Generate as ZRP model from input data & pre-trained pipeline. 
    
    Parameters
    ----------
    file_path: str, optional
        Path indicating where to put artifacts folder its files (pipeline, model, and supporting data), generated during intermediate steps.
    zrp_model_name: str
        Name of zrp_model
    zrp_model_source: str
        Indicates the source of zrp_modeling data to use. There are three optional sources 'bg' (for block_group), 'ct' (for census_tract), and 'zp' (for zip_code). By default 'census_tract' is inferred.
    """

    def __init__(self, file_path=None, zrp_model_name='zrp_0', zrp_model_source='ct', *args, **kwargs):
        super().__init__(file_path=file_path, *args, **kwargs)
        self.zrp_model_name = zrp_model_name
        self.zrp_model_source = zrp_model_source
        self.outputs_path = os.path.join(self.out_path,
                                         "experiments",
                                         self.zrp_model_name,
                                         self.zrp_model_source,
                                         "data")
        self.geo_key = 'GEOID'

    def fit(self, X, y):
        ### Build the zrp_model
        ##### specify zrp_model parameters
        print('\n---\nbuilding zrp_model')

        opt_params = {'gamma': 5,
                      'learning_rate': 0.01,
                      'max_depth': 3,
                      'min_child_weight': 500,
                      'n_estimators': 2000,
                      'subsample': 0.20}

        ##### Initialize the zrp_model
        self.zrp_model = XGBClassifier(objective='multi:softprob',
                                       num_class=len(y[self.race].unique()),
                                       **opt_params)
        ##### Fit
        print('\n---\nfitting zrp_model')
        self.zrp_model.fit(
            X, y[self.race],
            sample_weight=y.sample_weight
        )

        self.y_unique = y[self.race].unique()
        self.y_unique.sort()

        return self

    def transform(self, X):
        make_directory(self.outputs_path)
        # Save zrp_model
        pickle.dump(self.zrp_model, open(os.path.join(self.outputs_path, "zrp_model.pkl"), "wb"))
        try:
            self.zrp_model.save_model(os.path.join(self.outputs_path, "model.txt"))
        except:
            pass

        ##### Return Race Probabilities
        print('\n---\nGenerate & save race predictions (labels)')
        y_hat_train = pd.DataFrame({'race': self.zrp_model.predict(X)}, index=X.index)

        y_hat_train.reset_index(drop=False).to_feather(os.path.join(self.outputs_path, f"train_proxies.feather"))

        print('\n---\nGenerate & save race predictions (probabilities)')
        y_phat_train = pd.DataFrame(self.zrp_model.predict_proba(X), index=X.index)

        y_phat_train.columns = self.y_unique

        y_phat_train.reset_index(drop=False).to_feather(os.path.join(self.outputs_path, f"train_proxy_probs.feather"))

        return (y_hat_train, y_phat_train)


class ZRP_DataSampling(BaseZRP):
    """
    Generate data splits from input data 
    
    Parameters
    ----------
    file_path: str, optional
        Path indicating where to put artifacts folder its files (pipeline, model, and supporting data), generated during intermediate steps.
    zrp_model_name: str
        Name of zrp_model
    zrp_model_source: str
        Indicates the source of zrp_modeling data to use. There are three optional sources 'bg' (for block_group), 'ct' (for census_tract), and 'zp' (for zip_code). By default 'census_tract' is inferred.
    """

    def __init__(self, file_path=None, zrp_model_name='zrp_0', zrp_model_source='ct', *args, **kwargs):
        super().__init__(file_path=file_path, *args, **kwargs)
        self.zrp_model_name = zrp_model_name
        self.zrp_model_source = zrp_model_source
        self.outputs_path = os.path.join(self.out_path,
                                         "experiments",
                                         self.zrp_model_name,
                                         self.zrp_model_source,
                                         "data")
        self.geo_key = 'GEOID'

    def fit(self):
        return self

    def transform(self, data):
        make_directory(self.outputs_path)
        df = data.copy()

        # Keep geocoded data & data with labels
        #         df = df[(df[self.geo_key].notna()) & (df[self.geo_key]!="None")]
        df = df[(df[self.race].notna()) & (df[self.race] != "None")]
        df_keys = list(df.index.unique())

        # sample weights normalizing to us population
        aapi_ratio = df[self.race].value_counts(normalize=True).AAPI if 'AAPI' in df[self.race] else 1
        black_ratio = df[self.race].value_counts(normalize=True).BLACK if 'BLACK' in df[self.race] else 1
        hispanic_ratio = df[self.race].value_counts(normalize=True).HISPANIC if 'HISPANIC' in df[self.race] else 1
        aian_ratio = df[self.race].value_counts(normalize=True).AIAN if 'AIAN' in df[self.race] else 1
        other_ratio = df[self.race].value_counts(normalize=True).OTHER if 'OTHER' in df[self.race] else 1
        white_ratio = df[self.race].value_counts(normalize=True).WHIATE if 'WHIATE' in df[self.race] else 1

        df["sample_weight"] = df[self.race].map(
            {
                "AAPI": (0.061 / 1.022) / aapi_ratio,
                "BLACK": (0.134 / 1.022) / black_ratio,
                "HISPANIC": (0.185 / 1.022) / hispanic_ratio,
                "AIAN": (0.013 / 1.022) / aian_ratio,
                "OTHER": (0.028 / 1.022) / other_ratio,
                "WHITE": (0.601 / 1.022) / white_ratio,
            }
        )

        # Split working data
        df.reset_index(inplace=True)
        X = df.copy()
        X.drop([self.race, "sample_weight"], axis=1, inplace=True)

        if self.geo_key == df.index.name:
            y = df[[self.geo_key, self.race, "sample_weight"]]
        else:
            y = df[[self.key, self.geo_key, self.race, "sample_weight"]]

        # Train (80) + Test(20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9)

        save_feather(X_train, self.outputs_path, f"X_train.feather")
        save_feather(y_train, self.outputs_path, f"y_train.feather")
        save_feather(X_test, self.outputs_path, f"X_test.feather")
        save_feather(y_test, self.outputs_path, f"y_test.feather")

        return (X_train, X_test, y_train, y_test)


class ZRP_Build(BaseZRP):
    """
    This class builds a new custom ZRP model trained off of user input data. Supply standard ZRP requirements including name, address, and race to build your custom model-pipeline. Race & ethnicity probablities and labels are returned from this class. The pipeline, model, and supporting data is saved automatically to "./artifacts/experiments/{zrp_model_name}/{zrp_model_source}/data/" in the support files path defined.
    
    Parameters
    ----------
    file_path: str
        Path indicating where to put artifacts folder its files (pipeline, model, and supporting data), generated during intermediate steps.
    zrp_model_name: str
        Name of zrp_model.
    zrp_model_source: str
        Indicates the source of zrp_modeling data to use. There are three optional sources 'bg' (for block_group), 'ct' (for census_tract), and 'zp' (for zip_code). By default 'census_tract' is inferred.
    """

    def __init__(self, file_path=None, zrp_model_name='zrp_0', zrp_model_source='ct', *args, **kwargs):
        super().__init__(file_path=file_path, *args, **kwargs)
        self.zrp_model_name = zrp_model_name
        self.zrp_model_source = zrp_model_source
        self.outputs_path = os.path.join(self.out_path,
                                         "experiments",
                                         self.zrp_model_name,
                                         self.zrp_model_source,
                                         "data")
        self.geo_key = 'GEOID'

    def fit(self):
        return self

    def transform(self, data):
        make_directory(self.outputs_path)
        sample_path = self.outputs_path

        # Prepare data
        data = data.rename(columns = {self.first_name : "first_name", 
                              self.middle_name : "middle_name", 
                              self.last_name : "last_name",
                              self.house_number : "house_number", 
                              self.street_address : "street_address", 
                              self.city : "city",
                              self.zip_code : "zip_code",
                              self.state : "state", 
                              self.block_group : "block_group", 
                              self.census_tract : "census_tract"
                             }
                  )
        data = data.drop_duplicates(subset=['ZEST_KEY'])
        z_prepare = ZRP_Prepare(file_path=self.file_path)
        z_prepare.fit(data)
        prepared_data = z_prepare.transform(data)

        # Data Sampling 
        dsamp = ZRP_DataSampling(file_path=self.file_path)
        X_train, X_test, y_train, y_test = dsamp.transform(prepared_data)

        data = data.drop_duplicates(subset=['ZEST_KEY'])
        print("Post-sampling shape: ", data.shape)

        print("Unique labels: ", y_train['race'].unique())
        print("Other unique labels: ", y_test['race'].unique())
        cur_path = dirname(__file__)
        feature_list = load_json(os.path.join(cur_path, f'feature_list_{self.zrp_model_source}.json'))

        y_train = y_train.drop_duplicates(self.key)
        train_keys = list(y_train[self.key].values)
        X_train = X_train[X_train[self.key].isin(train_keys)]
        X_train = X_train.drop_duplicates(self.key)

        y_train[[self.geo_key, self.key]] = y_train[[self.geo_key, self.key]].astype(str)
        sample_weights = y_train[[self.key, 'sample_weight']].copy()

        assert X_train.shape[0] == y_train.shape[
            0], "Unexpected mismatch between shapes. There are duplicates in the data, please remove duplicates & resubmit the data"

        #### Set Index
        X_train.set_index(self.key, inplace=True)
        y_train.set_index(self.key, inplace=True)
        sample_weights.set_index(self.key, inplace=True)
        X_train.sort_index(inplace=True)
        y_train.sort_index(inplace=True)
        sample_weights.sort_index(inplace=True)

        feature_cols = list(set(X_train.columns) - set([self.key, self.geo_key, 'GEOID_BG', 'GEOID_CT',
                                                        'GEOID_ZIP', "first_name", "middle_name",
                                                        "last_name", 'ZEST_KEY_COL']))

        print('   train to numeric')
        X_train[feature_cols] = X_train[feature_cols].apply(pd.to_numeric, errors='coerce')

        print('\n---\nSaving raw data')
        save_feather(X_train, self.outputs_path, "train_raw_data.feather")
        save_feather(y_train, self.outputs_path, "train_raw_targets.feather")

        # Build Pipeline
        build_pipe = ZRP_Build_Pipeline(file_path=self.file_path)
        build_pipe.fit(X_train, y_train)
        X_train_fe = build_pipe.transform(X_train)

        # Build Model
        build_model = ZRP_Build_Model(file_path=self.file_path)
        build_model.fit(X_train_fe, y_train)
        y_hat_train, y_phat_train = build_model.transform(X_train_fe)

        pred_dict = {}
        pred_dict['labels'] = y_hat_train
        pred_dict['probablities'] = y_phat_train
        return (pred_dict)
