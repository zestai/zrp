from .preprocessing import *
from .base import BaseZRP
from .utils import *
import pandas as pd
import os
import warnings
warnings.filterwarnings(action='ignore')

def acs_search(year, span):
    """
    Searches for processed ACS data
    
    Parameters:
    -----------
    year: str
        Release year of ACS data
    span: str
        Span of ACS data (ie '1' or '5')      
    """
    file_list_z = []
    file_list_c = []
    file_list_b = []
    curpath = dirname(__file__)
    data_path = join(curpath, f'../data/processed/acs/{year}/{span}yr')
    for root, dirs, files in os.walk(os.path.join(data_path)):
        for file in files:
            if (f"_zip" in file) & ("processed" in file):
                file_list_z.append(os.path.join(root, file))
            if (f"tract" in file) & ("processed" in file):
                file_list_c.append(os.path.join(root, file))
            if (f"blockgroup" in file) & ("processed" in file):
                file_list_b.append(os.path.join(root, file))
    return (file_list_z, file_list_c, file_list_b)


class ACSModelPrep(BaseZRP):
    """
    Prepares ACS data & processed user input for modeling
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self):
        pass

    def acs_combine(self, data, acs_bg, acs_ct, acs_zip):
        """
        Combines ACS data with processed user input data.
        Generating optional features for modeling.
        
        Parameters
        ----------
        data: str
            Processed user input data, expected to include names & GEOID        
        acs_bg: str
            ACS block group lookup table
        acs_ct: str
            ACS census tract lookup table
        acs_zip: str
            ACS zip code lookup table
        """

        print(" ...Copy dataframes")
        ## Merge by current
        mbggk = data.copy()
        mctgk = data.copy()
        mbz = data.copy()
        nm = data.copy()

        print(" ...Block group")
        try:
            # Block Group
            mbggk_list = list(set(data.GEOID_BG.unique()).intersection(set(acs_bg.GEOID.unique())))        
            mbggk = mbggk[(mbggk.GEOID_BG.isin(mbggk_list))].reset_index(drop=False)
            mbggk_zkeys = list(mbggk.index)
            mbggk = mbggk.merge(acs_bg,
                                left_on="GEOID_BG",
                                right_on="GEOID").set_index(self.key)
            mbggk["acs_source"] = "BG"
        except AttributeError:
            mbggk_zkeys = []
            mbggk = pd.DataFrame()

        try:
        # Census Tract
            mctgk_list = list(set(data.GEOID_CT.unique()).intersection(set(acs_ct.GEOID.unique())))
            prev_zkeys = mbggk_zkeys

            mctgk = mctgk[~(mctgk.index.isin(prev_zkeys)) & \
                          (mctgk["GEOID_CT"].isin(mctgk_list))].reset_index(drop=False)
            mctgk_zkeys = list(mctgk.index)
            print(" ...Census tract")
            mctgk = mctgk.merge(acs_ct,
                                left_on="GEOID_CT",
                                right_on="GEOID").set_index(self.key)
            mctgk["acs_source"] = "CT"
        except AttributeError:
            mctgk_zkeys = []            
            mctgk = pd.DataFrame()
        try:
        # Merge by Zip
            mbz_list = list(set(data["GEOID_ZIP"].unique()).intersection(set(acs_zip.GEOID.unique())))
            prev_zkeys_0 = mbggk_zkeys + mctgk_zkeys
            print(" ...Zip code")

            mbz = mbz[~(mbz.index.isin(prev_zkeys_0)) & \
                      (mbz["GEOID_ZIP"].isin(mbz_list))].reset_index(drop=False)
            mbz_zkeys = list(mbz.index)
            mbz = mbz.merge(acs_zip,
                            right_on="GEOID",
                            left_on="GEOID_ZIP").set_index(self.key)
            mbz["acs_source"] = "ZIP"
        except AttributeError:
            mbz_zkeys = []
            mbz = pd.DataFrame()

        # No Merge
        print(" ...No match")
        prev_zkeys_1 = mbggk_zkeys + mctgk_zkeys + mbz_zkeys
        nm = nm[~(nm.index.isin(prev_zkeys_1))]
        nm["acs_source"] = None

        print(" ...Merge")
        data_out = pd.concat([mbggk, mctgk, mbz], sort=True)
        data_out = pd.concat([data_out, nm], sort=True)
        print(" ...Merging complete")
        return (data_out)

    def transform(self, input_data, save_table=False):
        """
        Parameters
        ----------
        input_data: pd.Dataframe
            Dataframe to be transformed
        save_table: bool
            Optional save
        """
        # Load Data
        try:
            data = input_data.copy()
        except AttributeError:
            data = load_file(input_data)
            print("Input data file is loaded")

        if "GEOID_ZIP" not in data.columns:
            print("Generating Geo IDs")
            data["GEOID_ZIP"] = data[self.zip_code]
            try:
                data["GEOID_CT"] = data[self.census_tract]
            except KeyError:
                pass
            try:
                data["GEOID_BG"] = data[self.block_group]
            except KeyError:
                pass

        file_list_z, file_list_c, file_list_b = acs_search(self.year,
                                                           self.span)

        print("   ...loading ACS lookup tables")
        acs_bg = load_file(file_list_b[0])
        acs_ct = load_file(file_list_c[0])
        acs_zip = load_file(file_list_z[0])

        print("   ... combining ACS & user input data")
        data_out = self.acs_combine(data,
                                    acs_bg,
                                    acs_ct,
                                    acs_zip)

        if save_table:
            make_directory(self.out_path)
            file_name = f"Zest_processed_data_.parquet"
            save_dataframe(data_out, self.out_path, file_name)
        return (data_out)
