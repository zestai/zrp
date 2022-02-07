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

def norm_na(data, na_values):
    """
    Standardize missing values
    
    Parameters
    ----------
    data: dataframe
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
    data: dataframe
        DataFrame to make changes to 
    key: str
        Key to set as index
    data_cols: list
        List of data columns available
    """
    data_cols = list(data.columns)
    if key == data.index.name:
        print("The key is already set")
    elif key in data_cols:
        data = data.set_index(key)
    else:
        data["tmp_key"] = data[data_cols].apply(lambda row:\
                                                "_".join(row.values.astype(str)),
                                                axis = 1)
        if data["tmp_key"].nunique() == len(data):
            data = data.rename(columns = {"tmp_key": key})
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
    data: dataframe
        DataFrame to make changes to 
    """
    return data.apply(lambda x: x.str.strip().str.replace(" +", " ", regex=True))


def replicate_address(data, i, street_suffix_mapping, unit_mapping):
    """
    Replicate street addresses 
    
    Parameters
    ----------
    data: dataframe
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
    """address mining"""
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
        data: dataframe
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

    
class  LongProcesStrings():
    """
    ProcesStrings executes all ZRP preprocessing. All user data is processed with additional  processing operations for geo-specific and American Community Survey data.
    
    
    Parameters
    ----------
    data: dataframe
        dataframe with user data
        
    state_mapping: dictionary
        dictionary mapping state names & abbreviations
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
    step:
        Indicator of step
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
    bisg=True
        Whether to return BISG proxies
    readout: bool
        Whether to return a readout
    n_jobs: int (default 1)
        Number of jobs in parallel
    """
    def __init__(self, key, first_name, middle_name, last_name, house_number, street_address, city, state, zip_code, support_files_path,  step, census_tract= None, street_address_2=None, name_prefix=None, name_suffix=None, na_values = None, file_path=None, geocode=True, bisg=True, readout=True, n_jobs=1):
        self.key = key
        self.first_name = first_name
        self.middle_name =  middle_name
        self.last_name = last_name
        self.name_suffix = name_suffix
        self.house_number = house_number
        self.street_address = street_address
        self.street_address_2 = street_address_2
        self.city = city
        self.state = state
        self.zip_code = zip_code
        self.census_tract = census_tract
        self.file_path = file_path
        self.support_files_path = support_files_path
        self.na_values = na_values
        self.geocode = geocode
        self.readout = readout
        self.step = step
        self.n_jobs = n_jobs
        super().__init__()
        
            
    def fit(self):
        pass
    
    
    def reduce_set(self, data):
        remove_cols = list(set(data.columns) - set([self.key,  self.first_name, self.middle_name, self.last_name, self.house_number, self.street_address, self.street_address_2, self.city, self.state, self.zip_code, self.census_tract]))
        return( data.drop(remove_cols, axis = 1))

    def transform(self, data_in):
        curpath = dirname(__file__)
        # Load Data
        try:
            data = data_in.copy()
            print("Data is loaded")
        except AttributeError:
            data = load_file(self.file_path)
            print("Data file is loaded")
            
        data_cols =  data.columns
        data = set_id(data, self.key)
        numeric_cols =  list(set([self.zip_code,
                                  self.census_tract,
                                  self.house_number
                                 ]).intersection(set(data_cols)))

        # Convert to uppercase & trim whitespace
        print("   Formatting P1")
        data = data.apply(lambda x: x.str.upper().str.strip())
        
        data = reduce_whitespace(data)
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
        if self.step=="geocoding":  #self.geocode:
            data_path = join(curpath, f'../data/processed')
            state_mapping, street_suffix_mapping, directionals_mapping, unit_mapping = load_mappings(data_path)
            print("   Geo Processing")
            street_addr_dict = dict(zip(data.index, data[self.street_address])) 
            street_addr_results = Parallel(n_jobs = self.n_jobs, prefer="threads", verbose=1)(delayed((address_mining))(street_addr_dict, i) for i in tqdm(list(data.index)))

            data[self.street_address] = street_addr_results

            data[self.city]  = data[self.city].str.replace("[^\\w\\s]", "", regex=True)

            # State
            data[self.state]  = data[self.state].str.replace("[^\\w\\s]", "", regex=True)
            data[self.state] = data[self.state].replace(state_mapping)
            # Zip code 
            data[self.zip_code] = np.where((data[self.zip_code].isna()) |\
                                     (data[self.zip_code].str.contains("None")),
                                      None,
                                      data[self.zip_code].apply(lambda x: x.zfill(5)))
            
            street_addr_dict = dict(zip(data.index, data[self.street_address])) 
            
            rep_addr_results = Parallel(n_jobs = self.n_jobs, prefer="threads", verbose=1)(delayed(replicate_address)(street_addr_dict, i, street_suffix_mapping, unit_mapping) for i in tqdm(list(data.index)))
            data[self.street_address] = rep_addr_results 
            
        if self.step=="glookup":
            print("   Lookup Processing")
            # State
            data["ZEST_STATE"] = data[self.state].replace(state_mapping)

            # Zip code 
            data["ZEST_ZIP"] = np.where((data["ZEST_ZIP"].isna()) |\
                                     (data["ZEST_ZIP"].str.contains("None")),
                                      None,
                                      data["ZEST_ZIP"].apply(lambda x: x.zfill(5)))
                    
        if self.step=="modeling":
            ht =  HandleTracts()
            data = ht.transform(data)
            data = data.replace({"\\bN\\b": None}, regex=True)
            if "ZEST_ZIP" in data.columns:
                data["ZEST_ZIP"] = np.where((data["ZEST_ZIP"].isna()) |\
                                         (data["ZEST_ZIP"].str.contains("None")),
                                          None,
                                          data["ZEST_ZIP"].apply(lambda x: x.zfill(5)))
            data = data.astype(str)
            
        print("   Formatting P2")
        data = norm_na(data, self.na_values)
        data = data.astype(str)
        data = reduce_whitespace(data)
        return data

    
def replicate_address_2(data, street_address, street_suffix_mapping, unit_mapping):
    """
    Replicate street addresses 
    
    Parameters
    ----------
    data: dataframe
       ACS dataFrame to make changes to 
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
    df_base =  data.copy()     # base is complete, containing the original record (1)
    print("         ...Map street suffixes...")
    data[street_address] = data[street_address].replace(street_suffix_mapping, regex=True) # this mapping takes the longest but is ok for 10K records
    print("         ...Mapped & split by street suffixes...")
    # Remove addtl "-"
    data[street_address] = data[street_address].str.split("-", 1, expand=True)
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
    print(f"     Address dataframe expansion is complete! (n={len(dataout)})")
    return(dataout)

       
class  ProcessStrings(BaseZRP):
    """
    ProcessStrings executes all ZRP preprocessing. All user data is processed with additional  processing operations for geo-specific and American Community Survey data.
    
    Parameters
    ----------
    data: dataframe
        dataframe with user data
        
    state_mapping: dictionary
        dictionary mapping state names & abbreviations
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
    bisg=True
        Whether to return BISG proxies
    readout: bool
        Whether to return a readout
    n_jobs: int (default 1)
        Number of jobs in parallel
    """
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
            
    def fit(self):
        pass
    
    
    def reduce_set(self, data):
        remove_cols = list(set(data.columns) - set([self.key,  self.first_name, self.middle_name, self.last_name, self.house_number, self.street_address, self.street_address_2, self.city, self.state, self.zip_code, self.census_tract]))
        return( data.drop(remove_cols, axis = 1))
    
    
    def transform(self, data_in, replicate=False):
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
        data = data.apply(lambda x: x.str.upper())
        
        data = reduce_whitespace(data)
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
        print("reduce whitespace")
        data = reduce_whitespace(data)
        return(data)
    
        
class  ProcessACS(BaseZRP):
    """
    ProcessStrings executes all ZRP preprocessing. All user data is processed with additional  processing operations for geo-specific and American Community Survey data.
    """
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
            
    def fit(self):
        pass


    def transform(self, data):
        """
        """
        data = reduce_whitespace(data)
        data = data.replace({'\\bN\\b': None}, regex=True) # Note: 800+ columns
        data = data.reset_index(drop=False)
        data = data.astype(str)
        return(data)
    
    
class  ProcessGeo(BaseZRP):
    """
    ProcessStrings executes all ZRP preprocessing. All user data is processed with additional  processing operations for geo-specific and American Community Survey data.
    
    
    Parameters
    ----------
    data: dataframe
        dataframe with user data
        
    state_mapping: dictionary
        dictionary mapping state names & abbreviations
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
    bisg=True
        Whether to return BISG proxies
    readout: bool
        Whether to return a readout
    n_jobs: int (default 1)
        Number of jobs in parallel
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
            
    def fit(self):
        pass


    def transform(self, data, processed, replicate=False):
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
            data = data.apply(lambda x: x.str.upper())
            data = reduce_whitespace(data)
            
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
        
        print("      ...address cleaning")
        street_addr_dict = dict(zip(data.index, data[self.street_address])) 
        street_addr_results = Parallel(n_jobs = self.n_jobs, prefer="threads", verbose=1)(delayed((address_mining))(street_addr_dict, i) for i in tqdm(list(data.index)))

        data[self.street_address] = street_addr_results
        data[self.city]  = data[self.city].str.replace("[^\\w\\s]", "", regex=True)

        # State
        data[self.state]  = data[self.state].str.replace("[^\\w\\s]", "", regex=True)
        data[self.state] = data[self.state].replace(state_mapping)
        # Zip code 
        data[self.zip_code] = np.where((data[self.zip_code].isna()) |\
                                 (data[self.zip_code].str.contains("None")),
                                  None,
                                  data[self.zip_code].apply(lambda x: x.zfill(5)))

        if replicate:
            print("      ...replicating address")
            data = replicate_address_2(data, self.street_address, street_suffix_mapping, unit_mapping)       
        
        print("      ...formatting")
        addr_cols = list(set(list(data.columns)).intersection(set([self.zip_code, self.census_tract, self.house_number, self.city, self.state, self.street_address])))
        spec_cols =  addr_cols
        data = data.astype(str)
        data = reduce_whitespace(data)
        print("   [Completed] Processing geo data")
        return(data)

    
class  ProcessGLookUp(BaseZRP):
    """
    ProcessStrings executes all ZRP preprocessing. All user data is processed with additional  processing operations for geo-specific and American Community Survey data.
    
    
    Parameters
    ----------
    data: dataframe
        dataframe with user data
        
    state_mapping: dictionary
        dictionary mapping state names & abbreviations
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
    bisg=True
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
        pass


    def transform(self, data, state_mapping):
        print("   [Start] Processing lookup data")
        # Load Data
        data_cols =  data.columns
        numeric_cols =  list(set([self.zip_code,
                                  self.census_tract,
                                  self.block_group,
                                  self.county,
                                  self.house_number
                                 ]).intersection(set(data_cols)))
        data = data.apply(lambda x: x.str.upper())
        data = reduce_whitespace(data)

        # Remove/replace special characters
        for col in numeric_cols:
            data[col] = data[col].apply(lambda x: re.sub("[^0-9]",\
                                                     "",\
                                                    str(x))) 

        # State
        data[self.state] = data[self.state].replace(state_mapping)
        data[self.zip_code] = np.where((data[self.zip_code].isna()) |\
                                 (data[self.zip_code].str.contains("None")),
                                  None,
                                  data[self.zip_code].apply(lambda x: x.zfill(5)))
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
        return(data)
    
