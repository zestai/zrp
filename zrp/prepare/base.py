import pandas as pd
import os


class BaseZRP():
    """
    Prepares data to generate race & ethnicity proxies
    
    Parameters
    ----------

    support_files_path: "data/processed"
        File path with support data.
    key: str, default 'ZEST_KEY', 
        Key to set as index. Optional. If not provided, a key will be generated.
    first_name: str, default 'first_name'
        Name of first name column
    middle_name: str, default 'middle_name'
        Name of middle name column
    last_name: str, default 'last_name'
        Name of last name/surname column
    house_number: str, default 'house_number'
        Name of house number column. Also known as primary address number this is the unique number assigned to a building to delineate it from others on a street. This is usually the first component of a delivery address line. Optional to not provide if Census GEOIDs are provided and no geocoding is required.
    street_address: str, default 'street_address'
        Name of street address column. The street address is usually comprised of predirectional, street name, and street suffix. 
    city: str, default 'city'
        Name of city column.
    state: str, default 'state'
        Name of state column.
    zip_code: str, default 'zip_code'
        Name of zip or postal code column.
    race: str, default 'race', optional
        Name of the race column
    census_tract: str, optional
        Name of Census tract column. Recommended to provide if available or a geocoder is typically implemented.
    block_group: str, optional
        Name of Census block group column. Recommended to provide if available or a geocoder is typically implemented.
    street_address_2: str, optional
        Name of additional address column
    name_prefix: str, optional
        Name of column containing full name preix (ie Dr, Sr, and Esq )
    name_suffix: str, optional
        Name of column containing full name suffix (ie jr, iii, and phd)
    na_values: list, optional
        List of missing values to replace.
    file_path: str, optional
        Path where to put artifacts and other files generated during intermediate steps. 
    geocode: bool, default True
        Geocoding indicator, to be deprecated by version 0.4.0.
    bisg: bool, default True
        Whether to return BISG proxies. 
    readout: bool, default True
        Whether to return a readout.
    n_jobs: int, default 49
        Number of jobs in parallel
    year: str, default '2019'
        ACS year to use.
    span: str, default '5'
        Year span of ACS data to use.
    runname: str, default None
        Name of the run.
    geocoding_type: str, default 'zrp'
        How to geocode the data. Possible methods: 'zrp', 'api', 'zrp->api', 'api->zrp'.
    censusapi_vintage: str, optional
        Edition of Census or Survey to be used while geocoding with Census API.
    """

    def __init__(self, support_files_path="data/processed", key="ZEST_KEY", first_name="first_name",
                 middle_name="middle_name", last_name="last_name", house_number="house_number",
                 street_address="street_address", city="city", state="state", zip_code="zip_code", race='race',
                 census_tract=None, block_group=None, street_address_2=None, name_prefix=None, name_suffix=None,
                 na_values=None, file_path=None, geocode=True, bisg=True, readout=True, n_jobs=-1, year="2020",
                 span="5", runname=None, geocoding_type='zrp', censusapi_vintage=None):
        self.key = key
        self.first_name = first_name
        self.middle_name = middle_name
        self.last_name = last_name
        self.name_suffix = name_suffix
        self.house_number = house_number
        self.street_address = street_address
        self.street_address_2 = street_address_2
        self.city = city
        self.state = state
        self.zip_code = zip_code
        self.census_tract = census_tract
        self.block_group = block_group
        self.file_path = file_path
        self.support_files_path = support_files_path
        self.na_values = na_values
        self.geocode = geocode
        self.readout = readout
        self.bisg = bisg
        self.race = race
        self.n_jobs = n_jobs
        self.year = year
        self.span = span
        self.runname = runname
        self.geocoding_type = geocoding_type
        if file_path:
            self.out_path = os.path.join(self.file_path, "artifacts")
        else:
            self.out_path = "artifacts"
        if censusapi_vintage is None:
            if int(self.year) >= 2020:
                self.censusapi_vintage = 'Census2020_Current'
            else:
                self.censusapi_vintage = 'ACS2019_Current'
        else:
            self.censusapi_vintage = censusapi_vintage

    def fit(self):
        pass

    def transform(self):
        pass

    def get_column_names(self):
        """
        Returns user defined column names of input DF. If user hasn't specified alternative names for columns, the default column names are returned.
        """
        column_names = []
        column_names.append(self.first_name)
        column_names.append(self.middle_name)
        column_names.append(self.last_name)
        column_names.append(self.house_number)
        column_names.append(self.street_address)
        column_names.append(self.city)
        column_names.append(self.state)
        column_names.append(self.zip_code)
        column_names.append(self.race)
        
        return(column_names)

    def reset_column_names(self):
        """
        Resets this class' stored column names to the defaults expected by ZRP. 
        """
        self.first_name = "first_name"
        self.middle_name = "middle_name"
        self.last_name = "last_name"
        self.house_number = "house_number"
        self.street_address = "street_address"
        self.city = "city"
        self.state = "state"
        self.zip_code = "zip_code"
