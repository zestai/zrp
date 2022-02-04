from zrp.prepare.utils import load_file
import pandas as pd
import numpy as np
import pycm
import os
import re


class ZRP_Performance():
    """
    Generates performance analysis artifacts
    
    
    Parameters
    ----------
    data: dataframe
        dataframe with user data
    key: str 
        Key to set as index. If not provided, a key will be generated.
    race: str
        Name of race column 
    proxy_data: str, pd.Series, or pd.DataFrame
        File path to proxy data
    ground_truth: str, pd.Series, or pd.DataFrame
        File path to ground truth data
    """

    def __init__(self, proxy_data, ground_truth, key = "ZEST_KEY", race="race"):
        self.key = key
        self.proxy_data = proxy_data
        self.ground_truth = ground_truth
        self.race = race
        
    def fit(self):
        return self
    
    def transform(self, input_data):
        # Load Data
        try:
            proxies = input_data.copy()
            print("Data is loaded")
        except AttributeError:
            proxies = load_file(self.proxy_data)
            print("Data file is loaded")
            
        try:
            ground_truth = input_data.copy()
            print("Data is loaded")
        except AttributeError:
            ground_truth = load_file(self.file_path)
            print("Data file is loaded")
        
        
        proxies = proxies[race]
        valid_proxies = valid_proxies[race]
            
        cm = ConfusionMatrix(
            np.array(ground_truth),
            np.array(proxies)
        )
        performance_dict = {}
        
        for metric in ["PPV", "TPR", "FPR", "FNR", "TNR", "AUC"]:
            performance_dict[metric] = None
            
        performance_dict['PPV'] = cm.PPV
        performance_dict['TPR'] = cm.TPR
        performance_dict['FPR'] = cm.FPR
        performance_dict['FNR'] = cm.FNR
        performance_dict['TNR'] = cm.TNR
        performance_dict['AUC'] = cm.AUC
        
        return(performance_dict)
        

