Why Is ZRP Breaking or Not Returning Proxies?
______________________________________________

In this document, we seek to provide insight for troubleshooting issues you may encounter while running the ZRP. We plan to address these issues in future ZRP updates. Contributions are encouraged! For small bug fixes and minor improvements, feel free to just open a PR. For larger changes, please open an issue first so that other contributors can discuss your plan, avoid duplicated work, and ensure it aligns with the goals of the project. Be sure to also follow the `Code of Conduct <https://github.com/zestai/zrp/blob/main/CODE_OF_CONDUCT.md>`_. Thanks!

#. A provided 'state' in the input dataset is an empty string. This causes an error when geocoding your data.
#. A provided 'state' in the input dataset is not in `inverse state mapping <https://github.com/zestai/zrp/blob/main/zrp/data/processed/inv_state_mapping.json>`_. The 'state' is required to be one of the 50 states in the US. We also use standard 2 letter abbreviations. 
#. ZRP has been run with only first name (or only middle name, last name, or address, respectively) provided. ZRP does not return a proxy when only one of these is provided; all must be a part of the input data frame.
#. There are spaces in the house number, ex.: "103 a". House numbers should not contain spaces.
#. Supplemental BISG proxies cannot be provided if zipcode is not a part of the input data.
#. You have duplicate rows in your data.
#. The current version of the ZRP accepts the following race "labels/classes": 'BLACK', 'HISPANIC', 'AAPI', 'AIAN', 'WHITE'. The input training data to ZRP_Build shouldn't contain any other labels in the provided 'race' column.

Manually Installing Lookup Tables and Pipeline Files
======================================================

#. Navigate to the `Latest Releases <https://github.com/zestai/zrp/releases>`_ page
#. Click on "lookup_tables.zip" and "pipelines.zip" for the latest release to download them locally
#. Unzip both of the zip folders
#. Navigate into the directory containing the pip installed zrp python package (referred to henceforth as ~/path_to_pip_installed_zrp)
#. Inside the unzipped "pipelines.zip" folder, you'll see the following
        | |----- block_group_pipe.pkl
        | |----- census_tract_pipe.pkl
        | |----- zip_code_pipe.pkl
    
    #. Inside your pip installed zrp directory will be the following subdirectories

        ~/path_to_pip_installed_zrp/modeling/models/block_group/

        ~/path_to_pip_installed_zrp/modeling/models/census_tract/

        ~/path_to_pip_installed_zrp/modeling/models/zip_code/
    #. Move block_group_pipe.pkl into **~/path_to_pip_installed_zrp/modeling/models/block_group/** AND once inside this directory, rename block_group_pipe.pkl to pipe.pkl 
    #. Move census_tract_pipe.pkl into **~/path_to_pip_installed_zrp/modeling/models/census_tract/** AND once inside this directory, rename census_tract_pipe.pkl to pipe.pkl 
    #. Move zip_code_pipe.pkl into **~/path_to_pip_installed_zrp/modeling/models/zip_code/** AND once inside this directory, rename zip_code_pipe.pkl to pipe.pkl 
    
#. Inside the unzipped "lookup_tables.zip" folder, you'll see the following folders
        | │   ├── acs
        | │   │   └── 2019
        | │   │       └── 5yr
        | │   │           ├── processed_Zest_ACS_Lookup_20195yr_blockgroup.parquet
        | │   │           ├── processed_Zest_ACS_Lookup_20195yr_tract.parquet
        | │   │           └── processed_Zest_ACS_Lookup_20195yr_zip.parquet
        | │   ├── geo
        | │   │   └── 2019
        | │   │       ├── Zest_Geo_Lookup_2019_State_01.parquet
        | │   │       ├── Zest_Geo_Lookup_2019_State_02.parquet
        | │   │       ├── Zest_Geo_Lookup_2019_State_04.parquet
        | │   │       ├── Zest_Geo_Lookup_2019_State_05.parquet
        | │   │       ├── .
        | │   │       ├── .
        | │   │       ├── .
    
    #. Move the acs/ and geo/ folders inside the **~/path_to_pip_installed_zrp/data/processed/** directory
