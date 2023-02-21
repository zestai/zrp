from pycm import ConfusionMatrix
from zrp.prepare.utils import load_file
from zrp.prepare.preprocessing import set_id
from sklearn.base import BaseEstimator, TransformerMixin 
from multiprocessing import Pool, Manager,cpu_count
from joblib import Parallel, delayed
from collections import defaultdict
from pycm import ConfusionMatrix
from sklearn import metrics 
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os
import re



class metrices:
    """
    Helper function to generate metrics on Multiple probability threshold    
    Parameters
    ----------
    data: data frame with zrp_output structure
    thres: threshold to apply on dataframe

    """

    def __init__(self,data,thres=0,target_col ='race',cols =None):
        self.data = data 
        self.thres = thres
        self.target_col = 'race'
        self.classes = data[self.target_col].unique()  
        self.filterd_df=  self._apply_threshold()
        
        
    def _apply_threshold(self):
        return self.data.loc[(self.data[self.data[self.target_col].unique()]>self.thres).max(axis=1)]
        
        
    def TPR(self):
        ## prob is a series containing the prediction probability for given class
        ## TPR is true positive/ actual positive
        tprd = defaultdict()
        act = self.filterd_df[self.target_col]
        prob= self.filterd_df[self.classes].idxmax(axis=1) 
        for col in self.classes:
            pred_onehot=  (prob == col).astype(int)
            act_onehot  = (act == col).astype(int)
            true_positive = np.dot(pred_onehot,act_onehot)
            act_positive = act_onehot.sum()
            tprd[col] = true_positive/act_positive
        return tprd
    
    def FPR(self):
        fprd = defaultdict()
        act = self.filterd_df[self.target_col]
        prob= self.filterd_df[self.classes].idxmax(axis=1) 
        for col in self.classes:
            pred_onehot=  (prob == col).astype(int)
            act_onehot  = (act != col).astype(int)
            true_positive = np.dot(pred_onehot,act_onehot)
            act_positive = act_onehot.sum()
            fprd[col] = true_positive/act_positive
        return fprd
    
    ##PPV(Precision or positive predictive value) Micro or weighted avg 
    def PPV(self):
        fppv = defaultdict()
        act  = self.filterd_df[self.target_col]
        prob = self.filterd_df[self.classes].idxmax(axis=1) 
        for col in self.classes:
            pred_onehot=  (prob == col).astype(int)
            act_onehot  = (act == col).astype(int)
            true_positive = np.dot(pred_onehot,act_onehot)
            prd_positive = pred_onehot.sum()
            fppv[col] = true_positive/prd_positive
        return fppv





    
    

class ZRP_Performance(BaseEstimator, TransformerMixin):
    """
    Generates performance analysis artifacts
    
    Parameters
    ----------
    df: pd.DataFrame
        dataframe contains results predicted probabilities and target column
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
        return results from pycm module
        if 2 then return results from both sklearn and pycm modules

        
    Returns:
    ----------
    results = .fit_transform(df)
    result['pycm_results']: Provides all pycm results
    result['sklern_results']: All Metrics calculated using sklearn library
    result['coverage_met']: A dictionary with all metrics at different probability threshold
           
     
        
    """
    def __init__(self,key="ZEST_KEY", target_col="race",save_path= None,return_res = 1):
        self.key = key
        self.target_col = target_col
        self.return_res = return_res
        self.save_path = save_path
        
    def fit(self,x=None,y= None):
        return self
    
    
    def calculate_cov_thres(self,outfile,threshold):
        total_lines = outfile.shape[0]
        cov_dic1 =defaultdict()
        filter_bool = ((outfile[['AAPI', 'AIAN', 'BLACK', 'HISPANIC','WHITE']]>threshold).sum(axis=1)>0)
        fil_outfile = outfile[filter_bool]
        coverage = filter_bool.sum()
        coverage_per = (coverage/total_lines)*100
        outfile.race_proxy == outfile.race
        accuracy = (fil_outfile.race_proxy == fil_outfile.race).sum()/fil_outfile.shape[0]
        clf = metrices(outfile,threshold)
        cm1 = ConfusionMatrix(fil_outfile.race.values,fil_outfile.race_proxy.values)
        cov_dic1[threshold]=[coverage,coverage_per,cm1.overall_stat['Overall ACC']*100,cm1.ACC_Macro*100,cm1.TNR_Micro*100,cm1.TPR_Micro*100,
                           cm1.FNR_Micro*100,cm1.FPR_Micro*100]+ [i*100 for i in clf.TPR().values()]+ [i*100 for i in clf.FPR().values()] + [i*100 for i in clf.PPV().values()]
        df_col = ['Coverage','Coverage_Percentage','Accuracy','Accuracy_Macro','TNR_Micro','TPR_Micro','FNR_Micro','FPR_Micro'] + [i+'_TPR'for i in clf.TPR().keys()]+ [i+ '_FPR' for i in clf.FPR().keys()]+ [i+ '_PPV' for i in clf.PPV().keys()]

    
        return pd.DataFrame(cov_dic1,index = df_col)
    
    

    
    def calculate_cov(self,outfile):
        print(f"starting processing in {int(max(cpu_count()/3,1))} threads" )
        com_df = Parallel(n_jobs=int(max(cpu_count()/3,1)), verbose=1, prefer='processes')(delayed(self.calculate_cov_thres)(outfile,chunk) for chunk in tqdm(np.array(range(0,20,1))/20))
        final_df= pd.concat(com_df,axis=1).T
        return final_df.to_dict()
    
    
    
    
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
        self.key = "ZEST_KEY"
        
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

        ### Getting coverge ralated metrices
        
        self.coverage_met = self.calculate_cov(proxy_data)
        
        
        proxy_data = proxy_data.dropna()
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

        
        
        
        del(self.cm.__dict__['predict_vector'],self.cm.__dict__['actual_vector'])
        final_result = {}
        final_result['pycm_results'] = self.cm.__dict__
        final_result['sklern_results'] =  self.__dict__['sk_cm']
        final_result['coverage_met']= self.coverage_met
        
    
        if self.save_path != None:
            
            # Serializing json
            json_object = json.dumps(final_result, indent=4)

            # Writing to sample.json
            with open(self.save_path  +"/test_results.json", "w") as outfile:
                outfile.write(json_object)
            
            self.cm.save_html(self.save_path  +"/test_results")
        if self.return_res ==1:
            return final_result
        else:
            return dff_sk,dff_cm