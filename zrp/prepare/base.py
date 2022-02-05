import pandas as pd
import os
class BaseZRP():
    """
    Prepares data to generate race & ethnicity proxies
    
    
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
    
   
    
    def __init__(self, support_files_path="data/processed", key="ZEST_KEY", first_name="first_name", middle_name="middle_name", last_name="last_name", house_number="house_number", street_address="street_address", city="city", state="state", zip_code="zip_code", race='race', proxy="probs", census_tract= None, street_address_2=None, name_prefix=None, name_suffix=None, na_values = None, file_path=None, geocode=True, bisg=True, readout=True, n_jobs=49, year="2019", span ="5", runname="test"):
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
        self.bisg=bisg
        self.proxy = proxy
        self.race= race
        self.n_jobs = n_jobs
        self.year= year
        self.span = span
        self.runname = runname
        if file_path:
            self.out_path = os.path.join(self.file_path, "artifacts")
        else:
            self.out_path = "artifacts"

        super().__init__()
    def fit():
        pass
    def transform():
        pass
