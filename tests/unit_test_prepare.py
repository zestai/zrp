import unittest
from os.path import dirname, join, expanduser
# from zrp.validate import *
from zrp.prepare.preprocessing import ProcessStrings
from zrp.prepare.geo_geocoder import ZGeo
# from zrp.prepare.acs_mapper import *
from zrp.prepare.base import BaseZRP
from zrp.prepare.utils import load_json
import pandas as pd
import numpy as np
import warnings
import glob
import json
import sys
import os
import re
from tqdm import tqdm
    



class testZRP_Prepare(unittest.TestCase):

    
    def test_BaseZRP(self):
        ## Loading sample data for testing
        from zrp.prepare.prepare import ZRP_Prepare
        prepare = ZRP_Prepare()
        ## Loading sample file
        sample_test_file =  pd.read_csv("unit_test_data/2022-nj-mayors-sample.csv")
        prepare.fit(sample_test_file)

        data = sample_test_file.copy()
        ## loading lookup table
        lookup_tables_config = load_json("unit_test_data/lookup_tables_config.json")
        ## loading inv_state_mapping file 
        inv_state_map = load_json("unit_test_data/inv_state_mapping.json")
        ## First level of cleaning
        gen_process = ProcessStrings(file_path=prepare.file_path)
        data = gen_process.transform(data)
        ## adding the column fips
        data['zest_in_state_fips'] = data[gen_process.state].replace(inv_state_map)

        ##Geo code testing starts here
        geocode = ZGeo(file_path=gen_process.file_path)

        geocode_out = [] 
        geo_grps = data.groupby([gen_process.state])

        geo_dict = {}
        for s, g in geo_grps:
            geo_dict[s] = g
        gdkys = list(geo_dict.keys())


        geo_out = [] 
        for s in tqdm(gdkys):
            print("   ... on state:", str(s))
            geo = inv_state_map[s].zfill(2)
            output = geocode.transform(geo_dict[s], geo, processed = True, replicate = True, save_table = True)
            geocode_out.append(output)
            break



           
    
if __name__ =='__main__':
    unittest.main()