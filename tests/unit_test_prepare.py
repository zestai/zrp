from zrp.prepare.preprocessing import ProcessStrings
from zrp.prepare.acs_mapper import acs_search,ACSModelPrep
from zrp.prepare.geo_lookup import GeoLookUpBuilder
from zrp.prepare.utils import load_json,load_file
from pandas.util.testing import assert_frame_equal
from os.path import dirname, join, expanduser
from zrp.prepare.acs_lookup import ACS_Parser
from zrp.validate import ValidateGeocoded
from zrp.prepare.geo_geocoder import ZGeo
from zrp.prepare.base import BaseZRP
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

    



class testZRP_Prepare(unittest.TestCase):

    ## Helper function
    def _copy_and_overwrite(self,from_path, to_path):
        if os.path.exists(to_path):
            shutil.rmtree(to_path)
        shutil.copytree(from_path, to_path)
    ## keeping the test to be  monolithic 
    # Testing # prepare.py file here
    def step1_prepare(self):

        if not os.path.exists('./zrp/data'):
            os.mkdir('./zrp/data')


        ### Moving all static files to zrp_data directory
        self._copy_and_overwrite('./tests/unit_test_data','../zrp/data/processed/')
        ### moving artifects files to zrp directory

        if not os.path.exists('../zrp/artifacts'):
            os.mkdir('../zrp/artifacts')

        

        self._copy_and_overwrite('./tests/unit_test_data/artifacts_static','../zrp/artifacts')
        print("all required files are now moved to ")

        self._copy_and_overwrite('../zrp/data/','./zrp/data')

        ## Loading sample data for testing
        from zrp.prepare.prepare import ZRP_Prepare
        prepare = ZRP_Prepare()
        ## Loading sample file
        sample_test_file =  load_file("tests/unit_test_data/2022-nj-mayors-sample.csv")
        prepare.fit(sample_test_file)

        data = sample_test_file.copy()
        ## loading lookup table
        lookup_tables_config = load_json("tests/unit_test_data/lookup_tables_config.json")
        ## loading inv_state_mapping file 
        inv_state_map = load_json("tests/unit_test_data/inv_state_mapping.json")
        ## First level of cleaning
        gen_process = ProcessStrings(file_path=prepare.file_path)
        data = gen_process.transform(data)
        ## adding the column fips
        data['zest_in_state_fips'] = data[gen_process.state].replace(inv_state_map)
        pd.set_option('display.max_columns',None)
        
        ##Geo code testing starts here
        geocode = ZGeo(file_path=gen_process.file_path)
        geocode_out = [] 
        geo_grps = data.groupby([gen_process.state])

        geo_dict = {}
        for s, g in geo_grps:
            geo_dict[s] = g
        gdkys = list(geo_dict.keys())


        geo_out = [] 
        for s in gdkys:
            geo = inv_state_map[s].zfill(2)
            print(geo_dict[s].dtypes)
            output = geocode.transform(geo_dict[s], geo, processed = True, replicate = True, save_table = True)
            geocode_out.append(output)
        if len(geocode_out) > 0:
            geo_coded = pd.concat(geocode_out)

        geo_coded_keys = list(geo_coded.ZEST_KEY_COL.values)
        data_not_geo_coded = data[~data.index.isin(geo_coded_keys)]
        geo_coded = pd.concat([geo_coded, data_not_geo_coded])

        
        if gen_process.block_group is not None and gen_process.census_tract is not None:
            geo_coded = geo_coded.drop([gen_process.block_group, gen_process.census_tract], axis = 1)
            geo_coded = geo_coded.merge(data[[gen_process.block_group, gen_process.census_tract]], right_index = True, left_index = True, how = 'left')
            geo_coded['GEOID_BG'] = np.where((geo_coded[gen_process.block_group].isna()) | (geo_coded[gen_process.block_group].str.contains("None") | (geo_coded[gen_process.block_group] == ''))
                                             ,geo_coded['GEOID_BG']
                                             ,geo_coded[gen_process.block_group])
            geo_coded['GEOID_CT'] = np.where((geo_coded[gen_process.census_tract].isna()) | (geo_coded[gen_process.census_tract].str.contains("None") | (geo_coded[gen_process.census_tract] == ''))
                                             ,geo_coded['GEOID_CT']
                                             ,geo_coded[gen_process.census_tract])
            geo_coded = geo_coded.drop([gen_process.block_group, gen_process.census_tract], axis = 1) 


        validate = ValidateGeocoded()
        validate.fit()
        acs_validator = validate.transform(geo_coded)


        file_list_z, file_list_c, file_list_b = acs_search(gen_process.year,
                                                           gen_process.span)

        print("   ...loading ACS lookup tables")
        acs_bg = load_file('tests/unit_test_data/processed_Zest_ACS_Lookup_20195yr_blockgroup.parquet')
        acs_ct = load_file('tests/unit_test_data/processed_Zest_ACS_Lookup_20195yr_tract.parquet')
        acs_zip = load_file('tests/unit_test_data/processed_Zest_ACS_Lookup_20195yr_zip.parquet')


        amp = ACSModelPrep()
        data_out = amp.acs_combine(geo_coded, acs_bg, acs_ct, acs_zip)

        # data_out.to_csv("tests/unit_test_data/unit_test_results.csv")
        self.static_data = pd.read_parquet("./tests/unit_test_data/unit_test_results.parquet.gzip") 

        ### checking if schema is correct here
        out_schema = data_out.dtypes.to_dict()
        for key in out_schema.keys():
            self.static_data [key] =  self.static_data[key].astype(out_schema[key])
        
        ## Final test on the dataframes
        assert_frame_equal(self.static_data, data_out)


    def step2_aceparser(self):
        # Support files path pointing to where the raw ACS data is stored
        support_files_path = "./tests/unit_test_data/ACS_data/"
        # Year of ACS data
        year = "2019"
        # Span of ACS data. The ACS data is available in 1 or 5 year spans. 
        span = "1"
        # State
        state_level = "al"
        # State County FIPs Code
        st_cty_code = "01001"
        ## Testing ace perser here
        acs_parse = ACS_Parser(support_files_path = support_files_path, year = year, span = span, state_level = state_level, n_jobs=1 )
        output = acs_parse.transform(save_table = False)
        
        ## Here I am generating the data for 148 
        self.assertEqual(list(output.keys())[0],'148')
        # output['148']['data'].to_parquet('./tests/unit_test_data/acs_parsed.parquet.gzip',compression='gzip')
        self.ace_parsed_static = pd.read_parquet('./tests/unit_test_data/acs_parsed.parquet.gzip')
        out_schema = output['148']['data'].dtypes.to_dict()
        for key in out_schema.keys():
            self.ace_parsed_static[key] =  self.ace_parsed_static[key].astype(out_schema[key])
        assert_frame_equal(self.ace_parsed_static, output['148']['data'])

       


    def step3_dataprepgeo(self):
        # Support files path pointing to where the raw tigerline shapefile data is stored
        support_files_path = "./tests/unit_test_data/geo_sample/"
        # Year of shapefile data
        year = "2019"
        # Geo level to build lookup table at
        st_cty_code = "01001"

        geo_build = GeoLookUpBuilder(support_files_path = support_files_path, year = year,output_folder_suffix='new')

        output = geo_build.transform(st_cty_code, save_table = False)
        ## Reading previous file
        self.GEO_static = pd.read_parquet(os.path.join(support_files_path,'processed/geo/2019_new/Zest_Geo_Lookup_2019_01001.parquet'))
        ### checking if schema is correct here
        out_schema = output.dtypes.to_dict()
        for key in out_schema.keys():
            self.GEO_static [key] =  self.GEO_static[key].astype(out_schema[key])
        
        ## Final test on the dataframes
        assert_frame_equal(self.GEO_static, output)


    def step4_remove_unwanted_file(self):
        if os.path.exists('./zrp/data'):
            shutil.rmtree('./zrp/data')
        if os.path.exists('../zrp/data/processed/'):
            shutil.rmtree('../zrp/data/processed/')
        if os.path.exists('../zrp/artifacts'):
            shutil.rmtree('../zrp/artifacts')
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
