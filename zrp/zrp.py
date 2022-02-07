from zrp.modeling.predict import BISGWrapper, ZRP_Predict
from os.path import dirname, join, expanduser
from zrp.prepare.prepare import ZRP_Prepare
from zrp.prepare.base import BaseZRP
from zrp.prepare.utils import *
import pandas as pd
import numpy as np
import warnings
import surgeo
import pickle
import joblib
import json
import pycm
import sys
import os
import re


class ZRP(BaseZRP):
    """Zest Race Predictor, predicts race & ethnicity using name & geograhpic data
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def fit(self):
        return self
    
    def transform(self, input_data):
        """
        Parameters
        ---------
        input_data: pd.Dataframe
            Dataframe to be transformed
        """
        # Load Data
        try:
            data = input_data.copy()
        except AttributeError:
            data = load_file(self.file_path)
        make_directory()
            
        z_prepare = ZRP_Prepare()
        z_prepare.fit(data)
        prepared_data =z_prepare.transform(data)
               
        if self.bisg:
            bisg_data = prepared_data.copy()
            bisg_data = bisg_data[~bisg_data.index.duplicated(keep='last')]
            
            bisgw = BISGWrapper()
            full_bisg_proxies = bisgw.transform(bisg_data)
            save_feather(full_bisg_proxies, self.out_path, f"bisg_proxy_{self.proxy}.feather")
            
        curpath = dirname(__file__)
        pipe_path = join(curpath, "modeling/models")
        
        z_predict = ZRP_Predict(pipe_path = pipe_path)
        z_predict.fit()
        predict_out = z_predict.transform(prepared_data)
        
        return(predict_out)
        
