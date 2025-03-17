import pandas as pd
import numpy as np
from zrp.prepare.utils import *


def is_missing(data, required_cols):
    """Checks if all required columns are provided
    
    Parameter
    ---------
    data: pd.dataframe
        dataframe to make changes to or use for validation
    required_cols: list
        list of required columns to check for
    """
    missing_columns = np.setdiff1d(required_cols,data.columns).tolist()
    return(missing_columns)


class BaseValidate():
    """
    This is a base validator which all validators inherit from.
    
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
    county: str, default 'county'   
        Name of the county column.        
    state: str
        Name of state column
    zip_code: str
        Name of zip or postal code column
    census_tract: str
        Name of census tract column
    block_group: str
        Name of block group column        
    race: str
        Name of race column
    year: str
        Name of year column
    span: str
        Name of span column
    """


    
    def __init__(self, support_files_path = "data/processed", key = "ZEST_KEY", first_name = "first_name", middle_name = "middle_name", last_name = "last_name", house_number = "house_number", street_address = "street_address", city = "city", county="county", state = "state", zip_code = "zip_code", race = "race", census_tract =  None, block_group = None, file_path = None, year = "2019", span  = "5"):
        self.key = key
        self.first_name = first_name
        self.middle_name =  middle_name
        self.last_name = last_name
        self.house_number = house_number
        self.street_address = street_address
        self.city = city
        self.county = county
        self.state = state
        self.zip_code = zip_code
        self.census_tract = census_tract
        self.block_group = block_group
        self.file_path = file_path
        self.race =  race
        self.year =  year
        self.span = span
        
    def fit(self, data):  
        data_cols = data.columns
        self.names = list(set(data_cols).intersection([self.first_name, self.middle_name, self.last_name]))
        self.geo_numerics = list(set(data_cols).intersection([self.zip_code, self.census_tract, self.block_group, self.house_number]))
        self.geo_strings = list(set(data_cols).intersection([self.street_address, self.city, self.state]))

        return self
        
        
    def check_row_counts(self, data):
        """Calculate row counts
        
        Parameter
        ---------
        data: pd.dataframe
            dataframe to make changes to or use for validation
        """
        return(len(data))
        
    def check_missing_pct(self, data, is_input = True):
        """Calculates percentage of missing values
        
        Parameter
        ---------
        data: pd.dataframe
            dataframe to make changes to or use for validation
        is_input: bool
            Indicator if validating raw input data
        
        Returns
        -------
        dict
            Dictionary with column names as keys and missing percentages as values.            
        """
        possible_zrp_cols =  set(data.columns).intersection({
            self.first_name, self.middle_name, self.last_name,
            self.house_number, self.street_address, self.county,
            self.state})
        na_dict = {}
        for col in possible_zrp_cols:
            if data[col].dtype == 'object':
                missing_pct = data[col].str.strip().str.upper().isin(['NONE', 'None', '', np.nan]).mean()
            else:
                missing_pct = data[col].isna().mean()
            
            na_dict[col] = missing_pct
    
            if missing_pct > 0.10:
                print(f"       (Warning!!) {col} is {missing_pct * 100:.2f}% missing")
        # Create an aggregate missing for geo ids
        potential_geo_ids = set(data.columns).intersection(set([self.block_group, "GEOID_BG", self.census_tract,"GEOID_CT", self.zip_code, 'GEOID_ZIP']))
        all_idx = set(data.index)
        num_idx= len(all_idx)
        for gcol in potential_geo_ids:
            all_idx = all_idx - set(data.index[data[gcol].notna()])
        na_dict['geoids'] = len(all_idx)/num_idx
        if na_dict['geoids'] > 0.10:
                print(f"       (Warning!!) geoids are {na_dict['geoids'] * 100:.2f}% missing")
        return na_dict

    
    def is_geocoded(self, data):
        """Calculates how much data is geocoded by geo-level
        
        Parameter
        ---------
        data: pd.dataframe
            dataframe to make changes to or use for validation
        """
        geocoded_cts = {}
        geocoded_cts["count"] = {}
        
        geocoded_cts["count"]["Block Group"] = (
            data["GEOID_BG"].str.len().gt(11).sum() if "GEOID_BG" in data else 0
        )
        geocoded_cts["count"]["Census Tract"] = (
            data["GEOID_CT"].str.len().gt(10).sum() if "GEOID_CT" in data else 0
        )
        geocoded_cts["count"]["Zip Code"] = (
            data["GEOID_ZIP"].str.len().eq(5).sum() if "GEOID_ZIP" in data else 0
        )
        return(geocoded_cts)
        
    def check_states(self, data):
        """Count of all states in user input data
                
        Parameter
        ---------
        data: pd.dataframe
            dataframe to make changes to or use for validation
        """
        return(data[self.state].value_counts(dropna = False).to_dict())
    

    def is_geoid(self, data, geoid_name):
        """Returns validation metrics, a length check to see if there is variation in geoid length typically look for zip to have 6 digits, census tract to have 11 digits (includes state and county code), and blockgroup to have atleast 12 digits.
                
        Parameter
        ---------
        data: pd.dataframe
            dataframe to make changes to or use for validation
        """
        geo_dict = {}
        geo_dict["length_check"] = data[geoid_name].str.len().value_counts(dropna = False).to_dict()
        geo_dict["numeric_check"] = bool(data[geoid_name].str.isnumeric().all())
        return(geo_dict)
        
    def is_mapped(self, data):
        """Determines how much data is mapped
                
        Parameter
        ---------
        data: pd.dataframe
            dataframe to make changes to or use for validation
        """
        mapped_dict = {}
        for acssrc in ["BG", "CT", "ZIP"]:
            mapped_sum = data[data["acs_source"] == acssrc].sum()
            mapped_dict [acssrc] = None
            mapped_dict[acssrc]["count"] = mapped_sum
            if mapped_sum>0:
                pct_mapped = mapped_sum/data.shape[0]
                mapped = True
            else:
                mapped_dict[acssrc]["percent"] = 0
        return(mapped_dict)
            
    def is_empty(self, data):
        """Checks if dataframe is empty
                
        Parameter
        ---------
        data: pd.dataframe
            dataframe to make changes to or use for validation
        """
        return(bool(data.empty))
    
    def is_all_missing(self, data):
        """Checks if all data in dataframe is missing
                
        Parameter
        ---------
        data: pd.dataframe
            dataframe to make changes to or use for validation
        """
        return(bool(data.isna().all().all()))

    def is_unique_key(self, data):
        """Checks if the provided key is unique
                
        Parameter
        ---------
        data: pd.dataframe
            dataframe to make changes to or use for validation
        """
        if self.key in data.columns:
            out = data[self.key].nunique() == data.shape[0]
        else:
            out = data.index.nunique() == data.shape[0]
        return(bool(out))
 
            
    def transform(self, data):
        pass
    

class ValidateGeo(BaseValidate):
    """
    Validates geo data
    
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
    county: str, default 'county'   
        Name of the county column.
    state: str
        Name of state column
    zip_code: str
        Name of zip or postal code column
    census_tract: str
        Name of census tract column
    block_group: str
        Name of block group column        
    race: str
        Name of race column
    year: str
        Name of year column
    span: str
        Name of span column
    """    
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.census_tract = 'GEOID_CT'
        self.block_group = 'GEOID_BG'
        self.zip_code = 'GEOID_ZIP'
        
    def fit(self):
        return self
            
    def transform(self, data):    
        validator = {}
        if self.zip_code in data.columns:
            validator["is_zip_code"] = self.is_geoid(data, self.zip_code)

        # convert to serializable json
        validator = convert_numpy(validator)

        return(validator)
        

class ValidateInput(BaseValidate):
    """
    Validates user input data
    
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
    county: str, default 'county'   
        Name of the county column.
    state: str
        Name of state column
    zip_code: str
        Name of zip or postal code column
    census_tract: str
        Name of census tract column
    block_group: str
        Name of block group column        
    race: str
        Name of race column
    year: str
        Name of year column
    span: str
        Name of span column
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def fit(self):
        return self
            
    def transform(self, data):
        validator = {}
        validator["is_empty"] = self.is_empty(data)
        if validator["is_empty"]:
            raise ValueError("Dataframe is empty")
        validator["is_all_missing"] = self.is_all_missing(data)
        if validator["is_all_missing"]:
            raise ValueError("Dataframe is fully missing")
        validator["n_obs"] = self.check_row_counts(data)
        print("     Number of observations:", validator["n_obs"])
        validator["is_unique_key"] = self.is_unique_key(data)
        print("     Is key unique:", validator["is_unique_key"]) 
        validator["pct_na"] = self.check_missing_pct(data)
        # convert to serializable json
        validator = convert_numpy(validator)        
        return(validator)
            
class ValidateGeocoded(BaseValidate):
    """
    Validates input ACS data that has geocoded keys.
    
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
    county: str, default 'county'   
        Name of the county column.
    state: str
        Name of state column
    zip_code: str
        Name of zip or postal code column
    census_tract: str
        Name of census tract column
    block_group: str
        Name of block group column        
    race: str
        Name of race column
    year: str
        Name of year column
    span: str
        Name of span column
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def fit(self):
        return self
            
    def transform(self, data):    
        validator = {}
        validator["is_empty"] = self.is_empty(data)
        if validator["is_empty"]:
            raise ValueError("Dataframe is empty")
        validator["is_all_missing"] = self.is_all_missing(data)
        if validator["is_all_missing"]:
            raise ValueError("Dataframe is fully missing")
        validator["n_obs"] = self.check_row_counts(data)
        print("     Number of observations:", validator["n_obs"])
        validator["is_unique_key"] = self.is_unique_key(data)
        print("     Is key unique:", validator["is_unique_key"]) 
                
        validator["pct_na"] = self.check_missing_pct(data, is_input=False)
        print("")
        try:
            for i in [self.last_name, self.first_name]:
                tmp = round(validator["pct_na"][i]*100, 2)
        except (KeyError, ValueError) as e:
            pass
        validator["is_geocoded"] = self.is_geocoded(data)
        
        # convert to serializable json
        validator = convert_numpy(validator)

        return(validator)        
    
    
class ValidateACS(BaseValidate):
    """
    Validates ACS data post mapping
    
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
    county: str, default 'county'   
        Name of the county column.        
    state: str
        Name of state column
    zip_code: str
        Name of zip or postal code column
    census_tract: str
        Name of census tract column
    block_group: str
        Name of block group column        
    race: str
        Name of race column
    year: str
        Name of year column
    span: str
        Name of span column
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def fit(self):
        return self
            
    def transform(self, data):    
        validator = {}
        validator["is_empty"] = self.is_empty(data)
        if validator["is_empty"]:
            raise ValueError("Dataframe is empty")
        validator["is_all_missing"] = self.is_all_missing(data)
        if validator["is_all_missing"]:
            raise ValueError("Dataframe is fully missing")
        validator["n_obs"] = self.check_row_counts(data)
        print("     Number of observations:", validator["n_obs"])
        validator["is_unique_key"] = self.is_unique_key(data)
        print("     Is key unique:", validator["is_unique_key"]) 
        validator["is_mapped"] = self.is_mapped(data)
        # convert to serializable json
        validator = convert_numpy(validator)
        return(validator)    
