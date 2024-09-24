from os.path import expanduser, dirname
import pandas as pd
import numpy as np
import os
import re
import sys
import json
import joblib
import pickle
import time

import xgboost
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
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
        Indicates the source of zrp_modeling data to use. There are three optional sources 'block_group', 'census_tract', and 'zip_code'. By default 'census_tract' is inferred.
    """

    def __init__(self, zrp_model_source, file_path=None, zrp_model_name='zrp_0', *args, **kwargs):
        super().__init__(file_path=file_path, *args, **kwargs)
        self.zrp_model_name = zrp_model_name
        self.zrp_model_source = zrp_model_source
        self.outputs_path = os.path.join(self.out_path,
                                         "experiments",
                                         self.zrp_model_name,
                                         self.zrp_model_source)
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

    def transform(self, X, file_name="train_fe_data.feather"):
        make_directory(self.outputs_path)
        # Save pipeline
        pickle.dump(self.pipe, open(os.path.join(self.outputs_path, 'pipe.pkl'), 'wb'))
        #### Transform
        ##### This step creates the feature engineering data
        print('\n---\nTransforming FE data')

        X_train_fe = self.pipe.transform(X=X)

        # Save train fe data
        print('\n---\nSaving FE data')
        if file_name is not None:
            save_feather(X_train_fe, self.outputs_path, file_name)
        return (X_train_fe)


def _weighted_multiclass_auc(pred, dtrain):
    """Used when custom objective is supplied."""
    y = dtrain.encoded_label
    weights = dtrain.get_weight()
    pred_softmax = np.exp(pred)
    pred_softmax = pred_softmax/(np.sum(pred_softmax,axis=1)[:,None])
    weighted_auc = 0.0
    weight_total = 0.0
    for iclass in range(pred.shape[1]):
        indx = np.where(y==iclass)[0]
        y_class = np.zeros(y.shape[0],dtype=np.int64)
        y_class[indx] = 1
        weight_total+=float(len(indx))
        weighted_auc+=float(len(indx))*roc_auc_score(y_class, pred_softmax[:,iclass])
    weighted_auc/=weight_total
    #return the negative since it tries to minimize the metric
    return 'WeightedAUC', -weighted_auc

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
        Indicates the source of zrp_modeling data to use. There are three optional sources 'block_group', 'census_tract', and 'zip_code'. By default 'census_tract' is inferred.
    xgb_params: dict (default=None)
        The xgboost model params to use when building the model.  If None then the default will be used 
        {'gamma': 5,'learning_rate': 0.01,'max_depth': 3,'min_child_weight': 500,'n_estimators': 2000,'subsample': 0.20}
    """
    class MultiClassDMatrix(xgboost.DMatrix):
        def __init__(self, data, label, *args, **kwargs):
            super(ZRP_Build_Model.MultiClassDMatrix,self).__init__(data, label=label, *args, **kwargs)
            self.encoded_label = label
            
    def __init__(self, zrp_model_source, file_path=None, zrp_model_name='zrp_0', xgb_params=None, *args, **kwargs):
        super().__init__(file_path=file_path, *args, **kwargs)
        self.zrp_model_name = zrp_model_name
        self.zrp_model_source = zrp_model_source
        self.outputs_path = os.path.join(self.out_path,
                                         "experiments",
                                         self.zrp_model_name,
                                         self.zrp_model_source)
        self.geo_key = 'GEOID'
        self.xgb_params = xgb_params

    def fit(self, X, y, X_valid=None, y_valid=None):
        ### Build the zrp_model
        ##### specify zrp_model parameters
        print('\n---\nbuilding zrp_model')
        print('\n training data shape:{},{}'.format(X.shape[0],X.shape[1]))
        
        if self.xgb_params is None:
            opt_params = {'gamma': 5,
                          'learning_rate': 0.01,
                          'max_depth': 3,
                          'min_child_weight': 500,
                          'n_estimators': 2000,
                          'subsample': 0.20,
                          'objective': 'multi:softprob'}
        else:
            opt_params = self.xgb_params.copy()
        objective = opt_params.pop('objective','multi:softprob')
        eval_metric = opt_params.pop('eval_metric','weighted_auc')
        if eval_metric=='weighted_auc' or eval_metric=='auc':
            feval=_weighted_multiclass_auc
            eval_metric=None
        elif callable(eval_metric):
            feval=eval_metric
            eval_metric=None            
        else:
            feval=None
            opt_params['eval_metric'] = eval_metric 
        early_stopping_rounds = opt_params.pop('early_stopping_rounds',None)  
        
        ##### Initialize the zrp_model
        label_encoder = xgboost.compat.XGBoostLabelEncoder().fit(y[self.race])
        y_dummies = label_encoder.transform(y[self.race])
        num_class=len(y[self.race].unique())
        self.zrp_model = XGBClassifier(objective=objective,
                                       num_class=num_class,
                                       **opt_params)
        tree_method = opt_params.pop('tree_method','auto')  
        num_boost_round=opt_params.pop('n_estimators',2000)
        if early_stopping_rounds is None:
            early_stopping_rounds = num_boost_round
        dtrain = ZRP_Build_Model.MultiClassDMatrix(X, y_dummies, weight=y.sample_weight)
        if X_valid is not None:
            y_valid_dummies = label_encoder.transform(y_valid[self.race])
            dvalid = ZRP_Build_Model.MultiClassDMatrix(X_valid, y_valid_dummies, weight=y_valid.sample_weight)
            evals = [(dtrain, 'train'), (dvalid, 'val')]
        else:
            evals = [(dtrain, 'train')]
            
        ##### Fit
        print('\n---\nfitting zrp_model... n_class={}'.format(num_class))
        #save_path = '/d/shared/users/gmw/zrp/model_artifacts'
        #save_feather(X, save_path, "fe_data_{}.feather".format(self.zrp_model_source))
        #save_feather(y[self.race], save_path, "target_data_{}.feather".format(self.zrp_model_source))    
        start_time = time.time()  # Start timing
        evals_result = dict()
        train_opts = {'objective':objective,'num_class':num_class,'tree_method':tree_method}
        if eval_metric is not None:
            train_opts['eval_metric'] = eval_metric
        
        model = xgboost.train(train_opts,
                          dtrain=dtrain,
                          num_boost_round=num_boost_round,
                          evals=evals,
                          early_stopping_rounds=early_stopping_rounds,
                          evals_result=evals_result,
                          feval=feval)  
        elapsed_time = time.time() - start_time
        self.y_unique = y[self.race].unique().astype(str)
        print('\n---\nfinished fitting zrp_model....{:.3f}'.format(elapsed_time))
        setattr(self.zrp_model,'classes_',self.y_unique)
        setattr(self.zrp_model,'n_classes_',num_class)
        setattr(self.zrp_model,'_le',label_encoder)    
        setattr(self.zrp_model,'_Booster',model)  
        setattr(self.zrp_model,'objective ',objective)
        setattr(self.zrp_model,'evals_result_ ',evals_result) 
        setattr(self.zrp_model,'best_score',model.best_score)  
        setattr(self.zrp_model,'best_iteration',model.best_iteration)   
        setattr(self.zrp_model,'best_ntree_limit',model.best_ntree_limit)
        
        self.y_unique.sort()
        
        make_directory(self.outputs_path)
        # Save zrp_model
        pickle.dump(self.zrp_model, open(os.path.join(self.outputs_path, "zrp_model.pkl"), "wb"))
        try:
            self.zrp_model.save_model(os.path.join(self.outputs_path, "model.txt"))
        except:
            pass
        print('\n---\nfinished saving zrp_model')
        return self

    def transform(self, X):

        ##### Return Race Probabilities
        print('\n---\nGenerate & save race predictions (labels)')
        y_hat_train = pd.DataFrame({'race': self.zrp_model.predict(X)}, index=X.index)

        y_hat_train.reset_index(drop=False).to_feather(os.path.join(self.outputs_path, f"train_proxies.feather"))

        print('\n---\nGenerate & save race predictions (probabilities)')
        y_phat_train = pd.DataFrame(self.zrp_model.predict_proba(X), index=X.index)

        y_phat_train.columns = self.y_unique

        y_phat_train.reset_index(drop=False).to_feather(os.path.join(self.outputs_path, f"train_proxy_probs.feather"))
        
        print("Artifacts saved to:", self.outputs_path)

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
        Indicates the source of zrp_modeling data to use. There are three optional sources 'block_group', 'census_tract', and 'zip_code'. By default 'census_tract' is inferred.
    population_weights_dict: dict
        Prevalence of target classes within the USA population as provided by the end-user. Sum of the values provided in the dictionary must be equal to one. Example: {'class1': 0.7, 'class2': 0.3}
    test_size: float (default=0.2)
        The fraction of samples to use as the test holdout
    valid_size: float (default=0.2)
        The fraction of samples to use as the test holdout
    """

    def __init__(self, zrp_model_source, file_path=None, zrp_model_name='zrp_0', population_weights_dict=None, test_size=0.2, valid_size=None, *args, **kwargs):
        super().__init__(file_path=file_path, *args, **kwargs)
        self.zrp_model_name = zrp_model_name
        self.zrp_model_source = zrp_model_source
        self.outputs_path = os.path.join(self.out_path,
                                         "experiments",
                                         self.zrp_model_name,
                                         self.zrp_model_source)
        self.geo_key = 'GEOID'
        self.population_weights_dict = population_weights_dict
        self.test_size = test_size
        self.valid_size = valid_size

    def fit(self):
        return self

    def transform(self, data):
        make_directory(self.outputs_path)
        df = data.copy()
        df = df[(df[self.race].notna()) & (df[self.race] != "None")]

        # sample weights normalizing to us population
        target_classes = list(df[self.race].unique())
        ratios = dict()
        for tc in target_classes:
            ratios[tc] = df[self.race].value_counts(normalize=True)[tc]
        
        sw_full_map = dict()
        for tc in target_classes:
            sw_full_map[tc] = np.round(self.population_weights_dict[tc]/ratios[tc] ,5)
    
        df["sample_weight"] = df[self.race].map(sw_full_map)

        # Split working data
        df.reset_index(inplace=True)
        X = df.copy()
        X.drop([self.race, "sample_weight"], axis=1, inplace=True)

        if self.geo_key == df.index.name:
            y = df[[self.geo_key, self.race, "sample_weight"]]
        else:
            y = df[[self.key, self.geo_key, self.race, "sample_weight"]]

        # Train (80) + Test(20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=9)
        
        X_valid = None
        y_valid = None
        if self.valid_size is not None:
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=self.valid_size/(1.0-self.test_size), random_state=19)
            save_feather(X_valid, self.outputs_path, f"X_valid.feather")
            save_feather(y_valid, self.outputs_path, f"y_valid.feather")      

        save_feather(X_train, self.outputs_path, f"X_train.feather")
        save_feather(y_train, self.outputs_path, f"y_train.feather")
        save_feather(X_test, self.outputs_path, f"X_test.feather")
        save_feather(y_test, self.outputs_path, f"y_test.feather")

        return (X_train, X_test, X_valid, y_train, y_test, y_valid)

class ZRP_Build(BaseZRP):
    """
    This class builds a new custom ZRP model trained off of user input data. Supply standard ZRP requirements including name, address, and race to build your custom model-pipeline. The pipeline, model, and supporting data is saved automatically to "./artifacts/experiments/{zrp_model_name}/{zrp_model_source}/" in the support files path defined.
    
    Parameters
    ----------
    file_path: str
        Path indicating where to put artifacts folder its files (pipeline, model, and supporting data), generated during intermediate steps.
    zrp_model_name: str
        Name of zrp_model.
    test_size: float (default=0.2)
        The fraction of samples to use as the test holdout
    valid_size: float (default=0.2)
        The fraction of samples to use as the test holdout
    xgb_params: dict (default=None)
        The xgboost model params to use when building the model.  If None then the default will be used 
        {'gamma': 5,'learning_rate': 0.01,'max_depth': 3,'min_child_weight': 500,'n_estimators': 2000,'subsample': 0.20}
    sources: list (default=None)
        The sources to build a model for.  If None is provided then all sources will be used: ['block_group', 'census_tract', 'zip_code']
    """
    def __init__(self, file_path=None, zrp_model_name='zrp_0', prepare_chunks_file_path=None, test_size=0.2, valid_size=None, xgb_params=None, sources=None, *args, **kwargs):
        super().__init__(file_path=file_path, *args, **kwargs)
        self.params_dict =  kwargs
        #self.z_prepare = ZRP_Prepare(file_path=self.file_path,  *args, **kwargs)
        self.zrp_model_name = zrp_model_name
        self.geo_key = 'GEOID'    
        if prepare_chunks_file_path is None:
            prepare_chunks_file_path = './prepare_chunks'
        self.prepare_chunks_file_path = prepare_chunks_file_path
        self.test_size = test_size
        self.valid_size = valid_size
        self.xgb_params = xgb_params
        self.sources = sources

    def validate_input_columns(self, data):
        """
        Passes if the input data has the requisite columns to run ZRP Build.
        
        Parameters
        -----------
        
        data: DataFrame
            A pandas data frame of user input data.
        """
        modeling_col_names = self.get_column_names
        for name in modeling_col_names():
            if name not in data.columns:
                raise KeyError("Your input dataframe has incorrect columns provided. Ensure that the following data is in your input data frame: first_name, middle_name, last_name, house_number, street_address, city, state, zip_code, race. If you have provided this data, ensure that the column names for said data are either the same as the aformentioned data column names, or ensure that you have specified, via arguements, the column names for these data you have provided in your input data frame.")
                
        return True
    
    def validate_target_classes(self, data, population_weights_dict, standard_population_weights_dicts):
        """
        Passes if the input data target classes are correctly specified.
        
        Parameters
        -----------
        
        data: DataFrame
            A pandas data frame of user input data.
        population_weights_dict: dict
            Prevalence of target classes within the USA population as provided by the end-user. Sum of the values provided in the dictionary must be equal to one. Example: {'class1': 0.7, 'class2': 0.3}
        standard_population_weights_dicts: list
            List of available dictionaries containing standard population weights
        """
        user_target_classes = set(data[self.race].unique())
        if population_weights_dict is None:
            matched_set_of_classes = 0
            for standard_population_weights in standard_population_weights_dicts:
                if user_target_classes == set(standard_population_weights.keys()):
                    matched_set_of_classes = 1
                    break
            if matched_set_of_classes == 0:
                raise ValueError(f'Non-standard set of target classes provided: \n\n\
...standard_sets = {[sorted(list(standard_population_weights.keys())) for standard_population_weights in standard_population_weights_dicts]} \n\n\
...provided = {sorted(list(user_target_classes))}\n\n\
"population_weights_dict" parameter must to specified to train on non-standard target classes')
        else:
            weights_classes = set(population_weights_dict.keys())
            if  weights_classes!= user_target_classes:
                raise ValueError(f'Dataset target classes and "population_weights_dict" target classes do not match')
            else:
                if sum(population_weights_dict.values()) != 1:
                    raise ValueError('Sum of "population_weights_dict" classes must be equal to 1')        

    def select_population_weights_dict(self, data, standard_population_weights_dicts):
        """
        Returns matching standard population weights dictionary.
        
        Parameters
        -----------
        
        data: DataFrame
            A pandas data frame of user input data.
        standard_population_weights_dicts: list
            List of available dictionaries containing standard population weights
        """
        user_target_classes = set(data[self.race].unique())
        for standard_population_weights in standard_population_weights_dicts:
            if user_target_classes == set(standard_population_weights.keys()):
                return standard_population_weights        
    
    def fit(self):
        return self

    def transform(self, data, population_weights_dict = None, chunk_size=25000):
        """
        Transforms the data
        
        Parameters
        -----------
        data: DataFrame
            A pandas data frame of user input data.
        population_weights_dict: dict
            Prevalence of target classes within the USA population as provided by the end-user. Sum of the values provided in the dictionary must be equal to one. Example: {'class1': 0.7, 'class2': 0.3}
        chunk_size: int
            Numer of rows to be processed in each iteration. input_data processed all at once if None is provided.
        """            
        cur_path = dirname(__file__)
        self.validate_input_columns(data)
        
        standard_population_weights_path = os.path.join(cur_path, '../data/processed/standard_population_weights.json')
        with open(standard_population_weights_path, 'r') as f:
            standard_population_weights_dicts = json.load(f)
        data["race"] = data["race"].str.replace(' ','_')
        data["race"] = data["race"].str.upper()
        self.validate_target_classes(data, population_weights_dict, standard_population_weights_dicts)
        if population_weights_dict is None:
            population_weights_dict = self.select_population_weights_dict(data, standard_population_weights_dicts)
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
                              self.census_tract : "census_tract",
                              self.race: "race"
                             }
                  )
        data = data.drop_duplicates(subset=['ZEST_KEY'])
        
        if chunk_size is None: 
            chunk_size = len(data) 
        chunk_max = int((len(data)-1)/chunk_size) + 1 
        prepare_out_list = list() 
        
        chunk_save_path = os.path.join(self.prepare_chunks_file_path,self.zrp_model_name)
        if not os.path.exists(chunk_save_path):
            print("Creating chunk path") 
            make_directory(chunk_save_path)
        
        existing_chunk_files = [fname for fname in os.listdir(chunk_save_path) if os.path.isfile(os.path.join(chunk_save_path, fname))]
        chunk_min = len(existing_chunk_files)
        if chunk_min>0:
            memory_loaded = 0.0
            print("####################################") 
            for chunk in range(np.min([chunk_min,chunk_max])):
                prepare_out_list.append(pd.read_feather(os.path.join(chunk_save_path,"prepare_chunk_{}.feather".format(chunk))))
                prepare_out_list[-1].set_index('ZEST_KEY',inplace=True)
                print(f'Loading processed rows: {chunk*chunk_size}:{(chunk+1)*chunk_size}'+',chunk_size = {}Mb'.format(np.sum(prepare_out_list[-1].memory_usage(deep=True))*1.0e-6)) 
                memory_loaded+=np.sum(prepare_out_list[-1].memory_usage(deep=True))*1.0e-6
            print('Total memory loaded = {:.3f}'.format(memory_loaded))
            print("####################################") 
        
        for chunk in range(chunk_min,chunk_max):  
            print("####################################") 
            print(f'Processing rows: {chunk*chunk_size}:{(chunk+1)*chunk_size}') 
            print("####################################") 
            data_chunk = data[chunk*chunk_size:(chunk+1)*chunk_size] 
            
            z_prepare = ZRP_Prepare(file_path=self.file_path, **self.params_dict) 
            z_prepare.fit(data_chunk)                                
            prepared_data_chunk = z_prepare.transform(data_chunk)
            save_feather(prepared_data_chunk, chunk_save_path, "prepare_chunk_{}.feather".format(chunk))
            print('chunk_size = {}Mb'.format(np.sum(prepared_data_chunk.memory_usage(deep=True))*1.0e-6))
            prepare_out_list.append(prepared_data_chunk) 
        print("Chunks being concatenated....") 
        prepared_data = pd.concat(prepare_out_list) 
        prepare_out_list = None 
        print("Finished chunks being concatenated....") 

        ft_list_source_map = {'census_tract': 'ct', 'block_group': 'bg', 'zip_code': 'zp'}
        source_to_geoid_level_map = {'census_tract': 'GEOID_CT', 'block_group': 'GEOID_BG', 'zip_code': 'GEOID_ZIP'}
        sources = ['block_group', 'census_tract', 'zip_code']
        if self.sources is not None:
            sources = self.sources if isinstance(self.sources,list) else [self.sources]
        
        for source in sources:
            print("=========================")
            print(f"BUILDING {source} MODEL.")
            print("=========================\n")
            outputs_path = os.path.join(self.out_path,
                                             "experiments",
                                             self.zrp_model_name,
                                             source)
            
            make_directory(outputs_path)
            
            # Get features to drop from prepared data
            print(f"Dropping {list(set(sources).difference({source}))} features")
            
            features_to_keep_list = load_json(os.path.join(cur_path, f'feature_list_{ft_list_source_map[source]}.json'))
            features_to_keep_list.append('race')
            
            print("    ...Len features to keep list: ", len(features_to_keep_list))
            
            # Get records that can be geocoded down to given source geo level
            geoid_level = source_to_geoid_level_map[source]
            relevant_source_data = prepared_data[~prepared_data[geoid_level].isna()]
            
            print("    ...Data shape pre feature drop: ", relevant_source_data.shape)
            relevant_source_data = relevant_source_data[relevant_source_data.columns.intersection(features_to_keep_list)]
            print("    ...Data shape post feature drop: ", relevant_source_data.shape)

            # Data Sampling 
            dsamp = ZRP_DataSampling(file_path=self.file_path, 
                                     zrp_model_source=source, 
                                     zrp_model_name=self.zrp_model_name,
                                     population_weights_dict = population_weights_dict,
                                     test_size=self.test_size,
                                     valid_size=self.valid_size)

            X_train, X_test, X_valid, y_train, y_test, y_valid = dsamp.transform(relevant_source_data)

            data = data.drop_duplicates(subset=['ZEST_KEY'])
            print("Post-sampling shape: ", data.shape)
            print("\n")
            print("Unique train labels: ", y_train['race'].unique())
            print("Unique test labels: ", y_test['race'].unique())

            y_train = y_train.drop_duplicates(self.key)
            train_keys = list(y_train[self.key].values)
            X_train = X_train[X_train[self.key].isin(train_keys)]
            X_train = X_train.drop_duplicates(self.key)

            y_train[[self.geo_key, self.key]] = y_train[[self.geo_key, self.key]].astype(str)
            sample_weights = y_train[[self.key, 'sample_weight']].copy()

            if X_train.shape[0] != y_train.shape[0]:
                raise AssertionError("Unexpected mismatch between shapes. There are duplicates in the data, please remove duplicates & resubmit the data")

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

            X_train[feature_cols] = X_train[feature_cols].apply(pd.to_numeric, errors='coerce')
            
            if X_valid is not None:
                y_valid = y_valid.drop_duplicates(self.key)
                valid_keys = list(y_valid[self.key].values)
                X_valid = X_valid[X_valid[self.key].isin(valid_keys)]
                X_valid = X_valid.drop_duplicates(self.key)

                y_valid[[self.geo_key, self.key]] = y_valid[[self.geo_key, self.key]].astype(str)

                if X_valid.shape[0] != y_valid.shape[0]:
                    raise AssertionError("Unexpected mismatch between shapes. There are duplicates in the data, please remove duplicates & resubmit the data")

                #### Set Index
                X_valid.set_index(self.key, inplace=True)
                y_valid.set_index(self.key, inplace=True)
                X_valid.sort_index(inplace=True)
                y_valid.sort_index(inplace=True)
                X_valid[feature_cols] = X_valid[feature_cols].apply(pd.to_numeric, errors='coerce')
                
            print('\n---\nSaving raw data')
            save_feather(X_train, outputs_path, "train_raw_data.feather")
            save_feather(y_train, outputs_path, "train_raw_targets.feather")

            # Build Pipeline
            build_pipe = ZRP_Build_Pipeline(file_path=self.file_path, zrp_model_source=source, zrp_model_name=self.zrp_model_name)
            build_pipe.fit(X_train, y_train)
            X_train_fe = build_pipe.transform(X_train)
            X_valid_fe = None
            if X_valid is not None:
                X_valid_fe = build_pipe.transform(X_valid, "valid_fe_data.feather")
                
            # Build Model
            build_model = ZRP_Build_Model(file_path=self.file_path, 
                                          zrp_model_source=source, 
                                          zrp_model_name=self.zrp_model_name,
                                          xgb_params=self.xgb_params)
            build_model.fit(X_train_fe, y_train, X_valid_fe, y_valid)
            
            print(f"Completed building {source} model.")
        
        print("\n##############################")
        print("Custom ZRP model build complete.")
        
