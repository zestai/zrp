import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin 


class Create_NGram(BaseEstimator, TransformerMixin):
    '''This class Create count based N-Gram Features. This script Extract N-Gram Features from the feature list
    Parameters
    ----------
    first_name: str
        Name of first name column
    middle_name: str
        Name of middle name column
    last_name: str
        Name of last name/surname column
    Ngram_feature_list: provide feature list 
        EX. a_count for count of "a" feature 
        ax_count for count of "ax" feature
    n_jobs: int (default 3)
        Number of jobs in parallel
    key: str
        Key to set as index
    ''' 
    
    def __init__(self, last_name, first_name, middle_name,Ngram_feature_list,n_jobs= 3,key = "ZEST_KEY"):
        self.last_name = last_name
        self.first_name = first_name
        self.middle_name = middle_name
        self.Ngram_feature_list = [i for i in Ngram_feature_list if "_count" in i]
        self.n_jobs = n_jobs
        self.key = key
        
    
    def fit(self,X,Y):
         return self
        
        
    def Create_Ngram(self,df,feature_name):
        feature_name = feature_name.replace("_count","")
        dt = df["Complete_Name"].apply(lambda x: x.count(f"{feature_name}"))
        dt.name = feature_name+"_count"
        return pd.DataFrame(dt)

    def transform(self, X):
        req_df = X[[self.key,self.last_name,self.first_name,self.middle_name]]
        req_df = req_df.set_index(self.key)
        
        req_df["Complete_Name"]= (req_df[self.first_name].astype(str)+" "+req_df[self.middle_name].astype(str)+" " +req_df[self.last_name].astype(str)).str.upper()
        req_df = req_df[["Complete_Name"]]
        
        
        ngram_results = Parallel(n_jobs = 3, prefer="threads", verbose=1)(delayed(self.Create_Ngram)(req_df, i) for i in tqdm(list(self.Ngram_feature_list)))
        
        return (X.merge(pd.concat(street_addr_results,1).reset_index(),on= self.key,how= 'left'))