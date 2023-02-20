from pycm import ConfusionMatrix
from zrp.prepare.utils import load_file
from zrp.prepare.preprocessing import set_id
from sklearn.base import BaseEstimator, TransformerMixin 
from pycm import ConfusionMatrix
from sklearn import metrics 
import pandas as pd
import numpy as np
import json
import os
import re

    
    

class ZRP_Performance(BaseEstimator, TransformerMixin):
    """
    Generates performance analysis artifacts
    
    Parameters
    ----------
    df: pd.DataFrame
        dataframe containes results predicted probabilities and target column
    key: str 
        default val = "ZEST_KEY"
        Key to set as index. If not provided, a key will be generated.
    target_col: str
        default val = "race"
        Name of race column 
    save_path: None
        "path to save final results in json and HTML format"
    return_result: int
        default val 1
        return results form pycm module
        if 2 then return results from both sklearn and pycm modules
        
    """
    def __init__(self,key="ZEST_KEY", target_col="race",save_path= None,return_res = 1):
        self.key = key
        self.target_col = target_col
        self.return_res = return_res
        self.save_path = save_path
        
    def fit(self,x=None,y= None):
        return self
    
    
    def calculate_tpr_fpr(self,y_real, y_pred):
        """
        Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations

        Args:
            y_real: The list or series with the real classes
            y_pred: The list or series with the predicted classes

        Returns:
            tpr: The True Positive Rate of the classifier
            fpr: The False Positive Rate of the classifier
       """

        # Calculates the confusion matrix and recover each element
        cm = metrics.confusion_matrix(y_real, y_pred)
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TP = cm[1, 1]

        # Calculates tpr and fpr
        tpr =  TP/(TP + FN) # sensitivity - true positive rate
        fpr = 1 - TN/(TN+FP) # 1-specificity - false positive rate
        fnr = FN/(TP + FN)
        tnr = TN/(FP+TN)

        PPV = TP/(TP+FP)
        ACC = (TP+TN)/cm.sum().sum()
        f1_score =  2*(PPV*tpr)/(PPV+tpr)
        COUNT = y_real.sum()
        



        return tpr,fpr,PPV,ACC,f1_score,fnr,tnr,COUNT



    def sklearn_results(self,proxy_data):
        
        sklearn_res = {}
        
        df = proxy_data
        if df.index.name != self.key:
            df.set_index(self.key,inplace= True)
        
        
        classes = ['WHITE', 'HISPANIC', 'BLACK', 'AIAN', 'AAPI']
        pred_proxy = df[classes].idxmax(1)  
        act = df[self.target_col]
            
            
        
        for i in range(len(classes)):
            mic_res ={}
            df_aux = pd.DataFrame()
            c = classes[i]
            df_aux['act'] = [1 if y == c else 0 for y in act]
            df_aux['pred'] = [1 if y == c else 0 for y in pred_proxy]

            md = self.calculate_tpr_fpr(df_aux['act'],df_aux['pred'])
            
            for count,met in enumerate(["TPR","FPR","PPV","ACC","F1","FNR","TNR","COUNT"]):
                mic_res[met]= md[count]
                

            fpr, tpr, thresholds = metrics.roc_curve(df_aux['act'].values, df[c].values, pos_label=1)
            mic_res['AUC'] = metrics.auc(fpr, tpr)
            sklearn_res[c]= mic_res
        
        sklearn_res = pd.DataFrame(sklearn_res).T.to_dict()
        
        
        return sklearn_res

    
    
    def pycm_results(self,proxy_data):
        """
        Returns confusion matrix analysis of ZRP performance against gound truth in the form of a dictionary
        Parameters
        ----------
        proxy_data: pd.dataframe
            Dataframe containing proxy race labels
        
        """
        ground_truth = proxy_data[[self.target_col]]
        proxies = proxy_data.copy()
        proxies = set_id(proxies, self.key)
        ground_truth = set_id(ground_truth, self.key)
        proxies = proxies[f"{self.target_col}_proxy"]
        ground_truth = ground_truth[self.target_col]
        
        cm = ConfusionMatrix(
            np.array(ground_truth),
            np.array(proxies)
        )
        performance_dict = {}
        return cm
    
    
    def transform(self, proxy_data):

        
        
        ### Getting sklearn based results
        self.sk_cm = self.sklearn_results(proxy_data)
        
        
        
        ### Getting pycm results here:
        
        self.cm = self.pycm_results(proxy_data)
        
        
        self.performance_dict_pycm = {}
        self.performance_dict_sklern = {}
        self.cm.COUNT = self.sk_cm['COUNT']

        for metric in ["PPV", "TPR", "FPR", "FNR", "TNR", "AUC","F1","COUNT","ACC"]:
            self.performance_dict_pycm[metric] = eval(f"self.cm.{metric}")
            self.performance_dict_sklern[metric] = eval("self.sk_cm['"+metric+"']")
            
        
        dff_sk  = pd.DataFrame(self.performance_dict_sklern).sort_index()
        dff_cm  = pd.DataFrame(self.performance_dict_pycm).sort_index()
        
#         for dff in [dff_sk,dff_cm]:
#             weighted_sum = [(dff[metric]*dff["COUNT"]).sum()/dff["COUNT"].sum() for metric in ["PPV", "TPR", "FPR", "FNR", "TNR", "AUC","F1","COUNT","ACC"]]
#             dff.loc['weighted_sum'] = weighted_sum
#             dff.loc['macro_avg'] = dff.mean()

        
    
        if self.save_path != None:
            del(self.cm.__dict__['predict_vector'],self.cm.__dict__['actual_vector'])
            final_result = {}
            final_result['pycm_results'] = self.cm.__dict__
            final_result['sklern_results'] =  self.__dict__['sk_cm']
            
            # Serializing json
            json_object = json.dumps(final_result, indent=4)

            # Writing to sample.json
            with open(self.save_path  +"/test_results.json", "w") as outfile:
                outfile.write(json_object)
            
            self.cm.save_html(self.save_path  +"/test_results")
        if self.return_res ==1:
            return dff_cm
        else:
            return dff_sk,dff_cm