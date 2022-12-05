# ZRP unit testing
## Contains all the details about unit testing framework

![N|Solid](data:image/jpeg;base64,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)



Steps to follow

- Create a new environment 
- Use pip to install all dependencies using pip install -r requirements.txt
- All unit test data should be moved to tests folder before trigring the pipeline
- unit testing data is availabe in shared folder on given path shared/zrp/unit_test_data.zip :-
- for testing the framework
--- cp shared/zrp/unit_test_data.zip zrp/tests/unit_test_data.zip
unzip the data 
--- unzip zrp/tests/unit_test_data.zip -d  zrp/tests

- Nevigate to zrp directory
- Run the unit test using "python -m tests.unit_test_prepare" 
- 

- ✨Magic ✨

## Features

- Is going to test all the modules used for data processing are working or not
- Is going to test wether the result generated by modules is correct or followes to the previous static release results


### Data Description 
- adding details about the data used for unit testing:


ACS_data directory: 
- Raw Data
1. A sample of 1 year American Community Survey data of 2019 for Alaska state,here are the name of files
    a. e20191ak0148000.txt: contians data
    b. e20191al0148000.txt
    c. seq148.xlsx : contais schema of data
    d. 1_year_Mini_Geo: geographical names and Geography ID for given state.
- Parsed Data
    a. Zest_ACS_nj_seq2_2019_1yr.parquet : A sample of processed 1-year American Community Survey data at the  level from the 2019 survey for state: NJ	GeoId: 04000US34	Geography Name:New Jersey
    b. processed_Zest_ACS_Lookup_20195yr_blockgroup.parquet:  processed 5-year American Community Survey data at the blockgroup level from the 2019 survey
    c. processed_Zest_ACS_Lookup_20195yr_tract.parquet: processed 5-year American Community Survey data at 
    tract the  level from the 2019 survey 
    d. processed_Zest_ACS_Lookup_20195yr_zip.parquet: processed 5-year American Community Survey data at the zip  level from the 2019 survey 

- Geo Data
  Zest_Geo_Lookup_2019_State_34.parquet: Geo lookup sample data for New Jersey
  state_mapping.json: a json that maps state names and state FIPS codes to standardized abbreviations
  inv_state_mapping.json: a json that maps standardized abbreviations for state to a number. 



  

processed_Zest_ACS_Lookup_20195yr_blockgroup.parquet: a sample of processed 5-year American Community Survey data at the block group level from the 2019 survey (what does the sample include? state or other geo location? number of records? etc.)


