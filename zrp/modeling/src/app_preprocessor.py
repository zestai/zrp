import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin 

class HandleCompoundNames(BaseEstimator, TransformerMixin):
    '''This class handles compound surnames.
    
    Parameters
    ----------
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
    ''' 
    

    def __init__(self, last_name, first_name, middle_name):
        self.last_name = last_name
        self.first_name = first_name
        self.middle_name = middle_name
        
    def fit(self, X, y):
        return self
    
    def _upper_case(self, X, columns):
        """Convert strings in the Series to uppercase.
        
        Parameters
        ----------
        X: dataframe
        columns: list
        """
        for col in columns:
            X[col] = X[col].str.upper()
        return X

    def _handle_compounds(self, X):
        """
        Simple compound surname handler. An individual with n names will receive a sample weight 1/n (where n = number of names).
        
        Parameters
        ----------
        X: dataframe
        """
        X[self.last_name] = X[self.last_name].str.replace('-', ' ', regex=False) 
        X[self.last_name] = X[self.last_name].str.replace(' +', ' ', regex=True)
        
        compound_name_str_all = X[self.last_name].str.split(' ', expand=True) # split compounds from non_compounds
        # Return if there're no compound names
        if compound_name_str_all.shape[1] == 1:
            print("Pass through")
            return X
        
        non_compound = X[compound_name_str_all[1].isna()].copy().reset_index(drop=True)
        compound = X[~compound_name_str_all[1].isna()].copy().reset_index(drop=True)
       
        compound_name_str = compound[self.last_name].str.split(' ', expand=True)
        n_compounds = compound_name_str.shape[1] # max number of unique strings
        compound_name_int = compound_name_str.copy() # binary representation of string
        for col in compound_name_str.columns:
            compound_name_int[col] = (~compound_name_str[col].isna()).astype(int)
        compound_name_int['compound_count'] = np.sum(compound_name_int, axis=1) #  row-wise count of strings  

        compound_tiled = compound.loc[compound.index.repeat(n_compounds)].reset_index(drop=True)
        sw_tiled = compound_name_int.loc[compound_name_int.index.repeat(n_compounds)].reset_index(drop=True)
        unrolled_names = (compound_name_str.values.ravel())
        compound_tiled[self.last_name] = unrolled_names
        compound_result = compound_tiled[~compound_tiled[self.last_name].isna()]

        joint_result = non_compound.append(compound_result).reset_index(drop=True)

        return joint_result

    def transform(self, X):
        data = X.copy()
        data.reset_index(drop=False, inplace=True)
        # compound names (row indicies are not preserved!)
        data = self._handle_compounds(data)
        return(data)