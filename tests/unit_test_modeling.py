from zrp.modeling.predict import ZRP_Predict
from zrp.prepare.utils import load_file, load_json
from zrp.zrp import ZRP
from tqdm import tqdm
import pandas as pd
import numpy as np
import unittest
import warnings
import shutil
import glob
import json
import sys
import os
import re


class testZRP_Modeling(unittest.TestCase):

    ## Helper function
    def _copy_and_overwrite(self,from_path, to_path):
        if os.path.exists(to_path):
            shutil.rmtree(to_path)
        shutil.copytree(from_path, to_path)
    ## keeping the test to be  monolithic 
    def step1_prepare(self):
        if not os.path.exists('./zrp/data'):
            os.mkdir('./zrp/data')
        if not os.path.exists('./zrp/data/processed'):
            os.mkdir('./zrp/data/processed')

        self._copy_and_overwrite('./tests/unit_test_data','./zrp/data/processed/')

        if not os.path.exists('./zrp/data/processed/geo'):
            os.mkdir('./zrp/data/processed/geo')
        if not os.path.exists('./zrp/data/processed/acs'):
            os.mkdir('./zrp/data/processed/acs')
        
        ## Moving the model .pkl files
        self._copy_and_overwrite('./tests/unit_test_data/geo','./zrp/data/processed/geo')
        self._copy_and_overwrite('./tests/unit_test_data/ACS_data/parsed/acs','./zrp/data/processed/acs')

        ## Rading base data for model test
        sample_test_file =  load_file("examples/2022-nj-mayors.csv")
        print( sample_test_file.head())
        zrp_sample = pd.DataFrame(columns=['first_name', 'middle_name', 'last_name', 'house_number', 'street_address', 'city', 'state', 'zip_code'])

        split_mayor_names = sample_test_file['MAYOR NAME'].str.split(' ')
        zrp_sample['first_name'] = split_mayor_names.str[0]
        zrp_sample['last_name'] = split_mayor_names.str[-1]
        zrp_sample['city'] = sample_test_file['CITY']
        zrp_sample['state'] =sample_test_file['STATE']
        zrp_sample['zip_code'] = sample_test_file['ZIP']
        zrp_sample['house_number'] = sample_test_file['ADDRESS 1'].str.extract('([0-9]+)')
        zrp_sample['street_address'] = sample_test_file['ADDRESS 1'].str.extract('.*[0-9]+([^0-9]+)')

        zrp_sample['ZEST_KEY'] = zrp_sample.index.astype(str) 

        if os.path.exists('./zrp/data/processed/artifacts'):
            shutil.rmtree('./zrp/data/processed/artifacts')

            
        zest_race_predictor = ZRP(file_path='./zrp/data/processed',
                                  pipe_path='./tests/unit_test_data/models')
        zest_race_predictor.fit()
        zrp_output = zest_race_predictor.transform(zrp_sample)



    def step2_artifect_input_validator(self):
        with open('./zrp/data/processed/artifacts/input_validator.json') as json_file:
            data = json.load(json_file)

        df = pd.DataFrame(data)
        json_file.close()
        self.assertEqual(df.shape,(7, 5))

    def step3_proxy_output(self):
        df = pd.read_feather("./zrp/data/processed/artifacts/proxy_output.feather")
        self.assertEqual(df.columns.to_list(),['ZEST_KEY',
        'AAPI',
        'AIAN',
        'BLACK',
        'HISPANIC',
        'WHITE',
        'race_proxy',
        'source_zrp_block_group',
        'source_zrp_census_tract',
        'source_zrp_zip_code',
        'source_bisg',
        'source_zrp_name_only'])
        self.assertEqual(df.shape,(565, 12))

    def step4_acs_validator(self):
        with open('./zrp/data/processed/artifacts/input_acs_validator.json') as json_file:
            data = json.load(json_file)

        df = pd.DataFrame(data)
        json_file.close()


        self.assertEqual(df.shape,(8, 6))

    def step5_geo_validator(self):
        with open('./zrp/data/processed/artifacts/input_geo_validator.json') as json_file:
             data = json.load(json_file)

        df = pd.DataFrame(data)
        json_file.close()
        self.assertEqual(df.shape,(2, 1))

     
    def step6_bisg_proxy(self):
        df = pd.read_feather("./zrp/data/processed/artifacts/bisg_proxy_output.feather")
        self.assertEqual(df.columns.to_list(),['ZEST_KEY',
        'AAPI',
        'AIAN',
        'BLACK',
        'HISPANIC',
        'WHITE',
        'race_proxy',
        'source_bisg'])
        self.assertEqual(df.shape,(565, 8))

    def step7_remove_unwanted_file(self):
        if os.path.exists('./zrp/data'):
            shutil.rmtree('./zrp/data')
        pass

    def _steps(self):
        for name in dir(self): # dir() result is implicitly sorted
            if name.startswith("step"):
                yield name, getattr(self, name) 


    def test_steps(self):
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
        



if __name__ =='__main__':
    unittest.main()

