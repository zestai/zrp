import pandas as pd
import numpy as np
import os
import re
import sys
from os.path import join, expanduser
import json
import warnings
warnings.filterwarnings(action='ignore')
 

def load_json(path):
    """
    Load json files
    
    Parameters
    ----------
    path: str
        Filepath of json file
    """
    with open(path, "r") as infile:
        data = json.load(infile)
    return data


def save_json(data, path, file_name):
    """
    Save json files
    
    Parameters
    ----------
    data: list or dict
        Data to save to json
    path: str
        Filepath of where to store json
    file_name: str
        Name of file
    """    
    with open(os.path.join(path, file_name), "w") as outfile:
        json.dump(data, outfile)


def save_dataframe(data, path, file_name):
    """
    Save dataframe as parquet
    
    Parameters
    ----------
    data: pd.DataFrame
        Dataframe to save
    path: str
        Filepath of where to store json
    file_name: str
        Name of file
    """      
    data.to_parquet(os.path.join(path, file_name))
    return (print("Output saved"))


def save_feather(data, path, file_name):
    """
    Save dataframe as feather file
    
    Parameters
    ----------
    data: pd.DataFrame
        Dataframe to save
    path: str
        Filepath of where to store json
    file_name: str
        Name of file
    """          
    data.reset_index(drop = False).to_feather(os.path.join(path,
                                                         file_name))
    return(print("Output saved"))
        
def make_directory(output_directory = None):
    """
    Creates a directory
    
    Parameters
    ----------
    output_directory: str
        Filepath or name of directory to create
    """      

    try:
        if output_directory == None:
            os.makedirs("artifacts")
        else:
            os.makedirs(os.path.join(output_directory))
    except FileExistsError:
        print("Directory already exists")
        pass

def load_file(file_path):    
    """
    Load files. Compatible with csv, text, feather, xlsx, and parquet
    
    Parameters
    ----------
    file_path: str
        File path of file to load
    """

    na_values = ["None",
                 "NAN",
                 "NONE",
                 " ",
                 "(X)",
                 "-",
                 "  ",
                 "   ",
                 "-666666666",
                 "-999999999",
                 "-888888888"
                 ]
    if file_path.endswith(".csv"):
        data = pd.read_csv(file_path,
                           dtype=str,
                           na_values=na_values)
    elif file_path.endswith(".feather"):
        data = pd.read_feather(file_path)
        data = data.astype(str)
    elif file_path.endswith(".xlsx"):
        data = pd.read_excel(file_path,
                             dtype=str,
                             na_values=na_values)
    elif file_path.endswith(".parquet"):
        data = pd.read_parquet(file_path)
        data = data.astype(str)
    elif file_path.endswith(".txt"):
        with open(file_path) as f:
            first_line = f.readline()
            if "|" in first_line:
                data = pd.read_csv(file_path,
                                   sep="|",
                                   dtype=str,
                                   na_values=na_values)
            elif "," in first_line:
                data = pd.read_csv(file_path,
                                   sep=",",
                                   dtype=str,
                                   na_values=na_values)
            else:
                data = pd.read_csv(file_path,
                                   sep="\t",
                                   dtype=str,
                                   na_values=na_values)
    else:
        raise ValueError("Unrecognizable table format")
    return (data)


def load_mappings(support_files_path):
    """
    Loads support mapping files
    
    Parameters
    ----------
    support_files_path: str
        Filepath or name of directory where files are stored
    """        
    mapping_file_path = os.path.expanduser(support_files_path)
    # add zip to county mapping
    state_mapping = load_json(os.path.join(mapping_file_path,
                                           "state_mapping.json"))
    street_suffix_mapping = load_json(os.path.join(mapping_file_path,
                                                   "street_suffix_mapping.json"))
    directionals_mapping = load_json(os.path.join(mapping_file_path,
                                                  "directionals_mapping.json"))
    unit_mapping = load_json(os.path.join(mapping_file_path,
                                          "unit_mapping.json"))
    return (state_mapping,
            street_suffix_mapping,
            directionals_mapping,
            unit_mapping)



def acs_rename(data):
    """
    Rename ACS index column
    
    Parameters
    -----------
    data: pd.DataFrame
        Dataframe to make changes to
    """
    data_cols = list(data.columns[1:])
    data.columns = ["result"] + data_cols
    return data


def acs_trt_split(data, feature):
    """
    Extract keys from ACS identifier column for census tract level data
    
    Parameters
    ----------
    data: pd.DataFrame
        Dataframe to make changes to
    feature: str
        Name of identifier column to extract information from
    """
    data = data[data[feature].notna()]
    state = [
        re.split(f".*state:", re.split(r">", data[feature].iloc[i])[0])[1]
        for i in range(len(data))
    ]

    county = [
        re.split(f".*county:", re.split(r">", data[feature].iloc[i])[1])[1]
        for i in range(len(data))
    ]

    tract = [
        re.split(f".*tract:", re.split(r">", data[feature].iloc[i])[2])[1]
        for i in range(len(data))
    ]

    data["STATEFP"] = state
    data["COUNTYFP"] = county
    data["TRACTCE"] = tract
    data["GEO_KEY"] = data["STATEFP"] + data["COUNTYFP"] + data["TRACTCE"]
    if "TRACTCE10" in data.columns:
        data["GEO_KEY_10"] = data["STATEFP"] + data["COUNTYFP"] + data["TRACTCE10"]
        data = data.drop("TRACTCE10", axis=1)
    data = data.drop([feature, "STATEFP", "COUNTYFP", "TRACTCE"], axis=1)

    return data


def acs_zip_split(dataframe, feature):
    """
    Extract keys from ACS identifier column for zip code level data
    
    Parameters
    ----------
    data: pd.DataFrame
        Dataframe to make changes to
    feature: str
        Name of identifier column to extract information from
    """    
    state = [
        re.split(f".*state:", re.split(r">", dataframe[feature].iloc[i])[0])[1]
        for i in range(len(dataframe))
    ]
    zcta5 = [
        re.split(f".*zip code tabulation area:", dataframe["result"].iloc[i])[1]
        for i in range(len(dataframe))
    ]
    dataframe["STATEFP"] = state
    dataframe["ZEST_ZIP"] = zcta5
    dataframe = dataframe.drop(
        ["Unnamed: 0", "result"],
        axis=1
    )


def most_common(lizt):
    """
    Returns the most common element of a list or series
    
    Parameters
    ----------
    lizt: list
        List to extract most common element from
    """
    return max(set(lizt), key=lizt.count)
