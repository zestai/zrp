from pycm import ConfusionMatrix

from zrp.prepare.utils import load_file
from zrp.prepare.preprocessing import set_id
from pycm import ConfusionMatrix
import pandas as pd
import numpy as np
import os
import re


class ZRP_Performance():
    """
    Generates performance analysis artifacts
    
    Parameters
    ----------
    key: str 
        Key to set as index. If not provided, a key will be generated.
    race: str
        Name of race column 
    proxy_data_path: str
        File path to proxy data
    ground_truth_path: str
        File path to ground truth data
    """

    def __init__(self, proxy_data_path=None,
                 ground_truth_path=None, key="ZEST_KEY", race="race"):
        self.key = key
        self.proxy_data_path = proxy_data_path
        self.ground_truth_path = ground_truth_path
        self.race = race

    def fit(self):
        return self

    def transform(self, proxy_data = None, ground_truth = None):
        """
        Returns confusion matrix analysis of ZRP performance against gound truth in the form of a dictionary

        Parameters
        ----------
        proxy_data: pd.dataframe
            Dataframe containing proxy race labels
        ground_truth: pd.dataframe
            Dataframe containing ground truth race labels
            """
        # Load Data
        try:
            proxies = proxy_data.copy()
        except AttributeError:
            proxies = load_file(self.proxy_data_path)
        try:
            ground_truth = ground_truth.copy()
        except AttributeError:
            ground_truth = load_file(self.ground_truth_path)

        proxies = set_id(proxies, self.key)
        ground_truth = set_id(ground_truth, self.key)
        proxies = proxies[f"{self.race}_proxy"]
        ground_truth = ground_truth[self.race]

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
