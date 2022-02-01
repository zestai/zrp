from .utils import *
from .base import ZRP
from .preprocessing import *
import pandas as pd
import numpy as np 
import os
import re
import sys
from os.path import join, expanduser
import json


def acs_search(support_files_path, year, span):
    file_list_z = []
    file_list_c = []
    file_list_b = []
    for root, dirs, files in os.walk(os.path.join(support_files_path, f"processed/acs/{year}/{span}yr")):
        for file in files:
            if (f"_zip" in file) & ("processed" in file) :
                file_list_z.append(os.path.join(root,file))
            if (f"tract" in file) & ("processed" in file) :
                file_list_c.append(os.path.join(root,file))
            if (f"blockgroup" in file) & ("processed" in file) :
                file_list_b.append(os.path.join(root,file))
    return(file_list_z, file_list_c, file_list_b)    



class ACSModelPrep(ZRP):
    """
    This class prepares ACS data & user input for modeling
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_data_path = os.path.join(self.support_files_path, f"processed/acs/{self.year}/{self.span}yr/dev") 
        #updated change back post dev
        
        
    def fit(self):
        pass
    

    def acs_combine(self, data, acs_bg, acs_ct, acs_zip):
        # Block Group
        mbggk_list = list(set(data.GEOID_BG.unique()).intersection(set(acs_bg.GEOID.unique())))
        print(data.index.name)
        
        print(" ...Copy dataframes") # new changed updated

        ## Merge by current
        mbggk = data.copy()
        mctgk = data.copy()
        mbz = data.copy()
        nm = data.copy()

        print(" ...Block group") # new changed updated
        mbggk = mbggk[(mbggk.GEOID_BG.isin(mbggk_list))].reset_index(drop=False)
        mbggk_zkeys = list(mbggk.index)

#             mbggk_zkeys = list(mbggk[self.key].unique())

        mbggk = mbggk.merge(acs_bg, left_on = "GEOID_BG", right_on = "GEOID").set_index(self.key)
        mbggk["acs_source"] = "BG"    

        # Census Tract
        mctgk_list = list(set(data.GEOID_CT.unique()).intersection(set(acs_ct.GEOID.unique())))

        ## Merge by current
        
        prev_zkeys = mbggk_zkeys
        mctgk = mctgk[~(mctgk.index.isin(prev_zkeys))  & (mctgk["GEOID_CT"].isin(mctgk_list))].reset_index(drop=False)
        
        mctgk_zkeys = list(mctgk.index)

#             mctgk_zkeys = list(mctgk[self.key].unique())        
        print(" ...Census tract") # new changed updated
        mctgk = mctgk.merge(acs_ct, left_on = "GEOID_CT", right_on = "GEOID").set_index(self.key)
        mctgk["acs_source"] = "CT" 


        # Merge by Zip
        mbz_list = list(set(data["GEOID_ZIP"].unique()).intersection(set(acs_zip.GEOID.unique())))

        prev_zkeys_0 = mbggk_zkeys + mctgk_zkeys
        print(" ...Zip code") # new changed updated
        
        mbz = mbz[~(mbz.index.isin(prev_zkeys_0))  & (mbz["GEOID_ZIP"].isin(mbz_list))].reset_index(drop=False)
        mbz_zkeys = list(mbz.index)

#             mbz_zkeys = list(mbz[self.key].unique())           

        mbz = mbz.merge(acs_zip, right_on ="GEOID", left_on="GEOID_ZIP").set_index(self.key)
        mbz["acs_source"] = "ZIP"

        # No Merge
        print(" ...No match") # new changed updated
        prev_zkeys_1 = mbggk_zkeys + mctgk_zkeys + mbz_zkeys 
        nm = nm[~(nm.index.isin(prev_zkeys_1))]
        

        print(" ...Merge") # new changed updated
        data_out = pd.concat([mbggk, mctgk, mbz], sort=True)#, ignore_index=True)

        data_out = pd.concat([data_out, nm], sort=True)#, ignore_index=True)
        print(" ...Merging complete") # new changed updated
        return(data_out)

    
    
    def transform(self, input_data, save_table=False):
        # Load Data
        try:
            data = input_data.copy()
            print("User input data is loaded")
        except AttributeError:
            data = load_file(input_data) 
            print("User input data file is loaded")
            
        if "GEOID_ZIP" not in  data.columns:
            print("Creating GEOIDs") # changed new updated
            data["GEOID_ZIP"] = data["ZEST_ZIP"]
            data["GEOID_CT"] = data["STATEFP"] + data["COUNTYFP"] + data["TRACTCE"] 
            data["GEOID_BG"] = data["GEOID_CT"] + data["BLKGRPCE"]
            
        file_list_z, file_list_c, file_list_b = acs_search(self.support_files_path, self.year, self.span)
        
        print("   ...loading ACS lookup tables")
        acs_bg = load_file(file_list_b[0])
        acs_ct = load_file(file_list_c[0])
        acs_zip = load_file(file_list_z[0])
        
        print("   ... combining ACS & user input data")
        
        data_out = self.acs_combine(data, acs_bg, acs_ct, acs_zip)
        
        if save_table:
            file_name = f"Zest_processed_data_dev_.parquet" # updated change back post dev
            save_dataframe(data_out, self.out_data_path, file_name)          
        return(data_out)
        
        
    
    
    
        
        
        