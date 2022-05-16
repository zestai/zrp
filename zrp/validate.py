import pandas as pd
import numpy as np


def is_missing(data, required_cols):
    """Checks if all required columns are provided
    Parameter
    ---------
    data: pd.dataframe
        dataframe to make changes to or use for validation
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


    
    def __init__(self, support_files_path = "data/processed", key = "ZEST_KEY", first_name = "first_name", middle_name = "middle_name", last_name = "last_name", house_number = "house_number", street_address = "street_address", city = "city", state = "state", zip_code = "zip_code", race = "race", census_tract =  None, block_group = None, file_path = None, year = "2019", span  = "5"):
        self.key = key
        self.first_name = first_name
        self.middle_name =  middle_name
        self.last_name = last_name
        self.house_number = house_number
        self.street_address = street_address
        self.city = city
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
        """
        data_cols = list(data.columns)
        possible_zrp_cols =  set(data_cols).intersection(set([
            self.first_name, self.middle_name, self.last_name,
            self.house_number, self.street_address, self.zip_code,
            self.state, self.block_group, self.census_tract,
        ]))
        na_dict = {}
        for col in data_cols:
            na_dict[col] = None
            na_dict[col] = data[(data[col].astype(str).str.upper() == "NONE")
                                | (data[col].astype(str).str.upper() == " ")
                                | (data[col].isna())].shape[0]/data.shape[0]
            if col in possible_zrp_cols:
                if na_dict[col] > 0.10:
                    print(f"       (Warning!!) {col} is {na_dict[col]*100}% missing, this may impact the ability to return race approximations")   
        return(na_dict)
    
    def is_geocoded(self, data):
        """Calculates how much data is geocoded by geo-level
        
        Parameter
        ---------
        data: pd.dataframe
            dataframe to make changes to or use for validation
        """
        geocoded_cts = {}
        geocoded_cts["count"] = {}
        
        geocoded_cts["count"]["GEOID"] = data[(data["GEOID"].str.len()>4)
                                                       & (data.index.duplicated(keep = "first"))].shape[0]
        geocoded_cts["count"]["Block Group"] = data[(data["GEOID_BG"].str.len()>11)  
                                                    & (data["GEOID_BG"] == data["GEOID"])
                                                    & (data["GEOID_BG"].notna())].shape[0]
        geocoded_cts["count"]["Census Tract"] = data[(data["GEOID_CT"].str.len()>10) 
                                                     & (data["GEOID_CT"] == data["GEOID"])
                                                     & (data["GEOID_CT"].notna())].shape[0]
        geocoded_cts["count"]["Zip Code"] = data[(data["GEOID_ZIP"].str.len() == 5)  
                                                 & (data["GEOID_ZIP"] == data["GEOID"])
                                                 & (data["GEOID_ZIP"].notna())].shape[0]
        return(geocoded_cts)
        
    def check_states(self, data):
        """Count of all states in user input data
                
        Parameter
        ---------
        data: pd.dataframe
            dataframe to make changes to or use for validation
        """
        return(data[self.state].value_counts(dropna = False).to_dict())
    
    def is_zcta5(self, data):
        """Determines if zip codes are provided as 5 digit zip codes
                
        Parameter
        ---------
        data: pd.dataframe
            dataframe to make changes to or use for validation
        """
        geo_dict = {}
        geo_dict["length_check"] = data[self.zip_code].str.len().value_counts(dropna = False).to_dict()
        geo_dict["numeric_check"] = bool(data[self.zip_code].str.isnumeric().all())
        return(geo_dict)
    
    def is_census_tract(self, data):
        """Determines if standard census tracts are provided
                
        Parameter
        ---------
        data: pd.dataframe
            dataframe to make changes to or use for validation
        """
        geo_dict = {}
        geo_dict["length_check"] = data[self.census_tract].str.len().value_counts(dropna = False).to_dict()
        geo_dict["numeric_check"] = bool(data[self.census_tract].str.isnumeric().all())
        return(geo_dict)
    
    def is_block_group(self, data):
        """Determines if standard block groups are provided
                
        Parameter
        ---------
        data: pd.dataframe
            dataframe to make changes to or use for validation
        """
        geo_dict = {}
        geo_dict["length_check"] = data[self.block_group].str.len().value_counts(dropna = False).to_dict()
        geo_dict["numeric_check"] = bool(data[self.block_group].str.isnumeric().all())
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
            data["acs_source"]
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
            validator["is_zip_code"] = self.is_zcta5(data)
        if self.is_census_tract in data.columns:
            validator["is_census_tract"] = self.is_census_tract(data)
        if self.is_block_group in data.columns:
            validator["is_block_group"] = self.is_block_group(data)
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
                assert tmp < 10, f"Too many missing values in required name feature, {i}. {tmp}% of the values are missing. Please review data and reduce missing. Required features include first name and last name." 
        except (KeyError, ValueError) as e:
            pass
        validator["is_geocoded"] = self.is_geocoded(data)
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
        return(validator)    