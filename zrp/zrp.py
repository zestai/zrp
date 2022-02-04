from os.path import join, expanduser
from zrp.prepare.utils import *
from zrp.prepare.prepare import *
from zrp.prepare.base import BaseZRP
from zrp.modeling.predict import BISGWrapper
import pandas as pd
import numpy as np
import warnings
import json
import sys
import os
import re
import pycm
import pickle
import joblib
import surgeo

class ZRP(BaseZRP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def fit(self):
        return self
    
    def transform(self, input_data):
        # Load Data
        try:
            data = input_data.copy()
        except AttributeError:
            data = load_file(self.file_path)
            
        z_prepare = ZRP_Prepare()
        z_prepare.fit(data)
        prepared_data =z_prepare.transform(data)
               
        if self.bisg:
            bisgw = BISGWrapper()
            full_bisg_proxies = bisgw.transform(prepared_data)
            save_feather(full_bisg_proxies, self.out_path, f"bisg_proxy_{self.proxy}.feather")
            
        pipe_path = "/d/shared/zrp/model_artifacts/experiment/exp_011"
        
        z_predict = ZRP_Predict(pipe_path = pipe_path)
        z_predict.fit()
        predict_out = z_predict(prepared_data)
        
        return(predict_out)
        