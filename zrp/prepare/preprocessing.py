from os.path import dirname, join, expanduser
from joblib import Parallel, delayed
from .base import BaseZRP
import multiprocessing
from tqdm import tqdm
from .utils import *
import pandas as pd
import numpy as np 
import json
import sys
import os
import re
from zrp.validate import *
import warnings
warnings.filterwarnings(action='ignore')

def norm_na(data, na_values):
    """
    Standardize missing values
    
    Parameters
    ----------
    data: pd.DataFrame
        DataFrame to make changes to 
    na_values: list
        List of missing values to replace
    """
    base_na_list = ["\\b#N/A\\b",
                    "\\b#N/A N/A\\b",
                    "\\b#NA\\b",
                    "\\b-1.#IND\\b",
                    "\\b-1.#QNAN\\b",
                    "\\b-NAN\\b",
                    "\\b-NaN\\b",
                    "\\b-nan\\b",
                    "\\b1.#IND\\b",
                    "\\b1.#QNAN\\b",
                    "\\b<NA>\\b",
                    "\\bN/A\\b",
                    "\\bNA\\b",
                    "\\bNULL\\b",
                    "\\bNAN\\b",
                    "\\bNaN\\b",
                    "\\bn/a\\b",
                    "\\bnan\\b",
                    "\\bnull\\b",
                    "\\bNULL\\b",
                    "\\bNONE\\b",
                    "^//(X//)$",
                    "^-$",
                   "^\\s*$"] 

    if na_values:
        word_border = "\\b"
        na_values = [word_border + s + word_border for s in na_values]
        na_values = na_values + base_na_list
    else:
        na_values = base_na_list

    na_dict = {}    
    for key in na_values:
        na_dict[key] = None

    data = data.replace(na_dict, regex=True)
    return(data)


def set_id(data, key):
    """
    Set Key
    
    Parameters
    ----------
    data: pd.DataFrame
        DataFrame to make changes to 
    key: str
        Key to set as index
    data_cols: list
        List of data columns available
    """
    if key == data.index.name:
        print("The key is already set")
    elif key in list(data.columns):
        data = data.set_index(key)
    else:
        data["tmp_key"] = data[list(data.columns)].apply(lambda row:\
                                                "_".join(row.values.astype(str)),
                                                axis = 1)
        if data["tmp_key"].nunique() == len(data):
            data = data.rename(columns = {"tmp_key": key})
            data = data.set_index(key)
        else:
            data = data.sort_values("tmp_key")
            data[key] = data["tmp_key"] + data.index.astype(str)
            data = data.set_index(key)
    return(data)


def reduce_whitespace(data):
    """
    Reduce whitespace
    
    Parameters
    ----------
    data: pd.DataFrame
        DataFrame to make changes to 
    """
    return data.apply(lambda x: x.str.strip().str.replace(" +", " ", regex=True))


def replicate_address(data, i, street_suffix_mapping, unit_mapping):
    """
    Replicate street addresses 
    
    Parameters
    ----------
    data: pd.DataFrame
       ACS dataFrame to make changes to 
    street_address: str
       Name of street address column 
    street_suffix_mapping: dict
       Dictionary with street mappings
    unit_mapping: dict
       Dictionary with building unit mapping   
    """
    df_base =  data[i]     # base is complete, containing the original record (1)

    data[i] = pd.DataFrame([{i: data[i]}]).replace(unit_mapping, regex=True).loc[0, i]
    df_u = data[i].strip()
    data[i] = pd.DataFrame([{i: data[i]}]).replace(street_suffix_mapping, regex=True).loc[0, i] # Second Unit & Street Suffix abbreviation mapping
    
    df_sus =  data[i]     # street & unit mapping + split after suffix (2)
    df_suu =  data[i]     # street & unit mapping + split before unit (3)
    df_sus = df_sus.split("-", 1)[0].strip()
    df_u = df_u.split("-", 1)[0]
    df_suu = df_suu.split("-", 1)[0].strip()
    data[i] = data[i].replace("-", "").strip()
 
    df_su =  data[i]      # suffix & unit mapped to abbreviations (4) 
    df_no =  data[i]      # remove all numbers not-attached to str (5)

    df_no = re.sub("^([0-9]{1,6}[A-Z]{1,3})", "", df_no)
    df_no = re.sub(" [0-9]{4,7} ", "", df_no)

    dataout = pd.DataFrame([df_base,
                           df_u,
                           df_su,
                           df_sus,
                           df_suu,
                           df_no
                          ])
    dataout.index = [i, f"{i}_u", f"{i}_su", f"{i}_sus", f"{i}_suu",f"{i}_no" ]
    dataout = dataout.drop_duplicates()
    return(dataout)


def address_mining(data, i):
    """
    Cleans street addresses

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame to make changes to 
    i: int
        Iteration variable
    """
    data[i]  = re.sub("[^A-Za-z0-9\\s]",
                                   "",
                                   re.sub("[^A-Za-z0-9']",
                                          " ",
                                          str(data[i])))
    return(data[i])


class HandleTracts():
    """
    HandleTracts parses the American Community Survey response format into state, county, and tracts.
    It will also create a GEO_KEY for merging with geocoded user data.
    """
    def __init__(self):
        super().__init__()
    
    def fit(self, data):
        pass
    
    def transform(self, data):
        """
        Parameters
        ----------
        data: pd.DataFrame
           ACS dataFrame to make changes to 
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a dataframe")
        acs_ct = data.copy()
        acs_zip = data.copy()
        
        acs_rename(acs_ct)
        acs_rename(acs_zip)
        
        acs_trt_split(acs_ct, feature="result")
        acs_zip_split(acs_zip, feature="result")
        
        return(acs_ct, acs_zip)

    
def replicate_address_2(data, street_address, street_suffix_mapping, unit_mapping):
    """
    Replicate street addresses 
    
    Parameters
    ----------
    data: pd.DataFrame
        DataFrame to make changes to 
    street_address: str
       Name of street address column 
    street_suffix_mapping: dict
       Dictionary with street mappings
    unit_mapping: dict
       Dictionary with building unit mapping   
    """
    # base
    data = data.reset_index(drop=False)
    print("         ...Base")
    df_base =  data.copy()# base is complete, containing the original record (1)
    print("         ...Map street suffixes...")
    data[street_address] = data[street_address].replace(street_suffix_mapping, regex=True) # this mapping takes the longest but is ok for 10K records
    data[street_address] = np.where(data[street_address]=='nan', None, data[street_address])
    print("         ...Mapped & split by street suffixes...")
    # Remove addtl "-"
    data[street_address] = data[street_address].str.split("-", 1, expand=True)[0]
    data[street_address] = data[street_address].str.replace(pat = "-", repl = "", regex=False)
    print("         ...Number processing...")
    data[street_address] = data[street_address].str.replace("^([0-9]{1,6}[A-Z]{1,3})", "", regex=True)
    data[street_address] = data[street_address].str.replace("^([0-9]{1,3} [A-Z]{2})", "", regex=True)
    data[street_address] = data[street_address].str.replace(" [0-9]{4,7} ", "", regex=True)
    print("")
    dataout = pd.concat([df_base,
                           data
                          ], axis=0)
    dataout = dataout.drop_duplicates()
    print(f"         Address dataframe expansion is complete! (n={len(dataout)})")
    return(dataout)


def replicate_house_number(data, house_number):
    """
    Replicate street addresses 
    
    Parameters
    ----------
    data: pd.DataFrame
        DataFrame to make changes to 
    house_number: str
       Name of street address column 
    """
    # base
    data = data.reset_index(drop=True)
    print("         ...Base")
    df_base =  data.copy()# base is complete, containing the original record (1)
    print("         ...Number processing...")
    data[house_number].apply(lambda x: re.sub("[^0-9]",\
                                                     "",\
                                                    str(x))) 
    dataout = pd.concat([df_base,
                           data
                          ], axis=0)
    dataout = dataout.drop_duplicates()
    print(f"         House number dataframe expansion is complete! (n={len(dataout)})")
    return(dataout)

       
class ProcessStrings(BaseZRP):
    """
    ProcessStrings executes all ZRP preprocessing. All user data is processed with additional  processing operations for geo-specific and American Community Survey data.
    
    Parameters
    ----------
    key: str 
        Key to set as index. If not provided, a key will be generated.
    first_name: str
        Name of first name column
    middle_name: str
        Name of middle name column
    last_name: str
        Name of last name/surname column
    house_number: str
        Name of house number column. Also known as primary address number this is the unique number assigned to a building to delineate it from others on a street. This is usually the first component of a delivery address line.
    street_address: str
        Name of street address column. The street address is usually comprised of predirectional, street name, and street suffix. 
    city: str
        Name of city column
    state: str
        Name of state column
    zip_code: str
        Name of zip or postal code column
    census_tract: str
        Name of census tract column
    support_files_path:
        File path with support data
    street_address_2: str, optional
        Name of additional address column
    name_prefix: str, optional
        Name of column containing full name preix (ie Dr, Sr, and Esq )
    name_suffix: str, optional
        Name of column containing full name suffix (ie jr, iii, and phd)
    na_values: list
        List of missing values to replace 
    file_path: str
        Input data file path
    geocode: bool
        Whether to geocode
    race: str
        Name of race column
    bisg: bool, default True
        Whether to return BISG proxies
    readout: bool
        Whether to return a readout
    n_jobs: int (default 1)
        Number of jobs in parallel
    """
        
    def __init__(self, file_path=None, *args, **kwargs):
        super().__init__(file_path=file_path, *args, **kwargs)
            
    def fit(self, data):
        data_cols = list(data.columns)
        print("   [Start] Validating input data")
        if self.census_tract in data_cols:
            self.required_cols = [self.first_name, self.middle_name, self.last_name, self.census_tract]
        elif (self.census_tract in data_cols) & (self.block_group in data_cols):
            self.required_cols = [self.first_name, self.middle_name, self.last_name, self.census_tract, self.block_group]
        elif (self.zip_code in data_cols) & (self.geocode==True):
            self.required_cols = [self.first_name, self.middle_name, self.last_name, self.zip_code, self.house_number, self.street_address, self.city, self.state]
        elif (self.zip_code in data_cols) & (self.geocode==False):
            self.required_cols = [self.first_name, self.middle_name, self.last_name, self.zip_code]
        val_na = is_missing(data, self.required_cols)
        if val_na:
            assert True, f"     Missing required data {val_na}"
        validate = ValidateInput()
        validate.fit()
        validators_in = validate.transform(data)
        make_directory(self.out_path)
        save_json(validators_in, self.out_path, "input_validator.json")
        print("   [Completed] Validating input data")
        print("")
        return self

    
    def transform(self, data_in, replicate=False):
        """
        Processes general data

        Parameters
        ----------
        data_in: pd.DataFrame
            DataFrame to make changes to 
        replicate: bool
            Indicator to process address components
        """
        
        # Load Data
        try:
            data = data_in.copy()
        except AttributeError:
            data = load_file(self.file_path)
        
        data_cols =  data.columns
        data = set_id(data, self.key)
        
        numeric_cols =  list(set([self.zip_code,
                                  self.census_tract,
                                  self.house_number
                                 ]).intersection(set(data_cols)))

        # Convert to uppercase & trim whitespace
        print("   Formatting P1")
        data = data.astype(str)
        data = data.apply(lambda x: x.str.upper())
        
        data = reduce_whitespace(data)
        na_dict =  {"^\\s*$": None,
                    "^NAN$": None,
                    "^NONE$": None}
        data = data.replace(na_dict, regex=True)     
        
        # Remove/replace special characters
        for col in numeric_cols:
            data[col] = data[col].apply(lambda x: re.sub("[^0-9]",\
                                                     "",\
                                                    str(x))) 
        if self.last_name:
            name_cols = list(set([self.first_name,
                                  self.middle_name,
                                  self.last_name]).intersection(set(data_cols))) 
            for col in name_cols:
                    data[col] =  data[col].apply(lambda x: re.sub("[^A-Za-z\\s]",
                                                              "",
                                                              re.sub("[^A-Za-z']",
                                                                     " ",
                                                                     str(x))))
        print("   Formatting P2")
        print("   reduce whitespace")
        data = reduce_whitespace(data)
        return(data)
    
        
class  ProcessACS(BaseZRP):
    """
    All user data is processed with additional processing operations for geo-specific and American Community Survey data.

    Parameters
    ----------
    out_path: str
        Path to store output artifacts
    """    
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
            
    def fit(self, data):
        data_cols = list(data.columns)
        print("   [Start] Validating input ACS data")
        validate = ValidateGeocoded()
        validate.fit()
        acs_validator = validate.transform(data)
        save_json(acs_validator, self.out_path, "input_acs_validator.json")
        print("   [Completed] Validating input ACS data")
        print("")
        
        return self


    def transform(self, data):
        """
        Processes ACS data 

        Parameters
        ----------
        data_in: pd.DataFrame
            DataFrame to make changes to 
        """
        data = reduce_whitespace(data)
        data = data.replace({'\\bN\\b': None}, regex=True) # Note: 800+ columns
        data = data.reset_index(drop=False)
        return(data)
    
    
class  ProcessGeo(BaseZRP):
    """
    All user data is processed with additional  processing operations for geo-specific and American Community Survey data.

    Parameters
    ----------
    key: str 
        Key to set as index. If not provided, a key will be generated.
    first_name: str
        Name of first name column
    middle_name: str
        Name of middle name column
    last_name: str
        Name of last name/surname column
    house_number: str
        Name of house number column. Also known as primary address number this is the unique number assigned to a building to delineate it from others on a street. This is usually the first component of a delivery address line.
    street_address: str
        Name of street address column. The street address is usually comprised of predirectional, street name, and street suffix. 
    city: str
        Name of city column
    state: str
        Name of state column
    zip_code: str
        Name of zip or postal code column
    census_tract: str
        Name of census tract column
    support_files_path:
        File path with support data
    street_address_2: str, optional
        Name of additional address column
    name_prefix: str, optional
        Name of column containing full name preix (ie Dr, Sr, and Esq )
    name_suffix: str, optional
        Name of column containing full name suffix (ie jr, iii, and phd)
    na_values: list
        List of missing values to replace 
    file_path: str
        Input data file path
    geocode: bool
        Whether to geocode
    race: str
        Name of race column
    bisg: bool, default True
        Whether to return BISG proxies
    readout: bool
        Whether to return a readout
    n_jobs: int (default 1)
        Number of jobs in parallel
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
            
    def fit(self, data):
        data_cols = list(data.columns)
        print("   [Start] Validating input geo data")
        if self.census_tract in data_cols:
            self.required_cols = [self.census_tract]
        elif (self.census_tract in data_cols) & (self.block_group in data_cols):
            self.required_cols = [self.census_tract, self.block_group]
        elif (self.zip_code in data_cols) & (self.geocode==True):
            self.required_cols = [self.zip_code, self.house_number, self.street_address, self.city, self.state]
        val_na = is_missing(data, self.required_cols)
        val_na = is_missing(data, self.required_cols)
        if val_na:
            assert True, f"Missing required data {val_na}"
        geo_validate = ValidateGeo()
        geo_validate.fit()
        geo_validators_in = geo_validate.transform(data)
        save_json(geo_validators_in, self.out_path, "input_geo_validator.json") 
        print("   [Completed] Validating input geo data")
        
        return self


    def transform(self, data, processed, replicate=False):
        """
        Processes geo data 

        Parameters
        ----------
        data: pd.DataFrame
            DataFrame to make changes to 
        processed: bool
            Indicator is the data was previously cleaned & processed 
        replicate: bool
            Indicator to process compenents of the address
        """        
        curpath = dirname(__file__)
        print("   [Start] Processing geo data")
        # Load Data
        if not processed:
            data_cols =  data.columns
            data = set_id(data, self.key)
            
            numeric_cols =  list(set([self.zip_code,
                                      self.census_tract,
                                      self.house_number
                                     ]).intersection(set(data_cols)))
            

            print("      ...formatting")
            data = data.astype(str)
            data = data.apply(lambda x: x.str.upper())
            
            data = reduce_whitespace(data)
            na_dict =  {"^\\s*$": None,
                       "^NAN$": None,
                       "^NONE$": None}
            data = data.replace(na_dict, regex=True)
            # Remove/replace special characters
            for col in numeric_cols:
                data[col] = data[col].apply(lambda x: re.sub("[^0-9]",\
                                                         "",\
                                                        str(x))) 
            if self.last_name:
                name_cols = list(set([self.first_name,
                                      self.middle_name,
                                      self.last_name]).intersection(set(data_cols))) 
                for col in name_cols:
                        data[col] =  data[col].apply(lambda x: re.sub("[^A-Za-z\\s]",
                                                                  "",
                                                                  re.sub("[^A-Za-z']",
                                                                         " ",
                                                                         str(x))))
        
        data_path = join(curpath, f'../data/processed')
        state_mapping, street_suffix_mapping, directionals_mapping, unit_mapping = load_mappings(data_path)
        
        if self.key in data.columns:
            data["ZEST_KEY_COL"] = data[self.key]
        else:
            data["ZEST_KEY_COL"] = data.index        
        
        print("      ...address cleaning")
        street_addr_dict = dict(zip(data.index, data[self.street_address])) 
        street_addr_results = Parallel(n_jobs = self.n_jobs, prefer="threads", verbose=1)(delayed((address_mining))(street_addr_dict, i) for i in tqdm(list(data.index)))

        data[self.street_address] = street_addr_results
#         data[self.city]  = data[self.city].str.replace("[^\\w\\s]", "", regex=True)

        # State
        data[self.state]  = data[self.state].str.replace("[^\\w\\s]", "", regex=True)
        data[self.state] = data[self.state].replace(state_mapping)
        # Zip code 
        data[self.zip_code] = np.where((data[self.zip_code].isna()) |\
                                 (data[self.zip_code].str.contains("None")),
                                  None,
                                  data[self.zip_code].apply(lambda x: x.zfill(5)))
        data[self.zip_code] = data[self.zip_code].astype(str).str[:5]        
        
        if replicate:
            print("      ...replicating address")
            data = replicate_address_2(data, self.street_address, street_suffix_mapping, unit_mapping)
            data = replicate_house_number(data, self.house_number)
        
        print("      ...formatting")
        addr_cols = list(set(list(data.columns)).intersection(set([self.zip_code, self.census_tract, self.house_number, self.city, self.state, self.street_address])))


        data = reduce_whitespace(data)
        print("   [Completed] Processing geo data")
        return(data)

    
class  ProcessGLookUp(BaseZRP):
    """
    All user data is processed with additional  processing operations for geo-specific and American Community Survey data.
    Parameters
    ----------
    key: str 
        Key to set as index. If not provided, a key will be generated.
    first_name: str
        Name of first name column
    middle_name: str
        Name of middle name column
    last_name: str
        Name of last name/surname column
    house_number: str
        Name of house number column. Also known as primary address number this is the unique number assigned to a building to delineate it from others on a street. This is usually the first component of a delivery address line.
    street_address: str
        Name of street address column. The street address is usually comprised of predirectional, street name, and street suffix. 
    city: str
        Name of city column
    state: str
        Name of state column
    zip_code: str
        Name of zip or postal code column
    census_tract: str
        Name of census tract column
    support_files_path:
        File path with support data
    street_address_2: str, optional
        Name of additional address column
    name_prefix: str, optional
        Name of column containing full name preix (ie Dr, Sr, and Esq )
    name_suffix: str, optional
        Name of column containing full name suffix (ie jr, iii, and phd)
    na_values: list
        List of missing values to replace 
    file_path: str
        Input data file path
    geocode: bool
        Whether to geocode
    race: str
        Name of race column
    bisg: bool, default True
        Whether to return BISG proxies
    readout: bool
        Whether to return a readout
    n_jobs: int (default 1)
        Number of jobs in parallel
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.street_address= "ZEST_FULLNAME"
        self.state = "STATEFP"
        self.zip_code = "ZEST_ZIP"
        self.census_tract = "TRACTCE" 
        self.block_group = "BLKGRPCE"
        self.county = "COUNTYFP"
        
            
    def fit(self):
        return self


    def transform(self, data, state_mapping):
        """
        Processes geo lookup table data 

        Parameters
        ----------
        data: pd.DataFrame
            DataFrame to make changes to 
        state_mapping: dict
            Mapping of state codes and names to standard abbreviations 
        """               
        print("   [Start] Processing lookup data")
        # Load Data
        data_cols =  data.columns
        numeric_cols =  list(set([self.zip_code,
                                  self.census_tract,
                                  self.block_group,
                                  self.county,
                                 ]).intersection(set(data_cols)))
        data = data.astype(str)
        data = data.apply(lambda x: x.str.upper())
        
        data = reduce_whitespace(data)
        na_dict =  {"^\\s*$": None,
                    "^NAN$": None,
                    "^NONE$": None} 
        data = data.replace(na_dict, regex=True)
        
        # Remove/replace special characters
        for col in numeric_cols:
            data[col] = data[col].apply(lambda x: re.sub("[^0-9]",\
                                                     "",\
                                                    str(x))) 

      
        data[self.census_tract] = np.where((data[self.census_tract].isna()) |\
                                 (data[self.census_tract].str.contains("None")),
                                  None,
                                  data[self.census_tract].apply(lambda x: x.zfill(6)))
        data[self.block_group] = np.where((data[self.block_group].isna()) |\
                                 (data[self.block_group].str.contains("None")),
                                  None,
                                  data[self.block_group])
        data[self.county] = np.where((data[self.county].isna()) |\
                                 (data[self.county].str.contains("None")),
                                  None,
                                  data[self.county].apply(lambda x: x.zfill(3))) 
        
        print("     ...processing")
        spec_cols = list(set(list(data_cols)).intersection(set([self.zip_code, self.block_group, self.county, self.state, self.census_tract])))
        data[spec_cols] = norm_na(data[spec_cols], self.na_values)
        data = reduce_whitespace(data)
        print("   [Completed] Processing lookup data")
        
        data_cols = list(data.columns)
        validate = ValidateInput()
        validate.fit()
        validator_in = validate.transform(data) 
        print(validator_in)
        return(data)
