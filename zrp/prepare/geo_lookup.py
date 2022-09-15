from zrp.prepare.preprocessing import *
from os.path import dirname, join, expanduser
from zrp.prepare.utils import *
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import fiona
import json
import sys
import os
import re

def gdbToDf_short(file, indx):
    """
    Loads shapefiles
    
    Parameters
    ----------
    file: str
        Filepath or name of file to load
    indx: str
        Indicates type of shapefile to load in
    """      
    dictList = []
    if indx == "address":
        with fiona.open(file) as src:
            for i in range(10000000):
                dictList.append(next(src)["properties"])
    else:
        with fiona.open(file) as src:
            for i in range(len(src)):
                dictList.append(next(src)["properties"])
    df = pd.DataFrame(
        dictList,
        columns=dictList[0].keys(),
    )
    return df

def sides_split(addr_edge_df):
    """Split tigerline dataframes into sides
    Parameters
    ----------
    addr_edge_df: pd.DataFrame/tigerline dataframe with address features & edges data
    """
    l_addr_edge_df = addr_edge_df.copy()
    r_addr_edge_df = addr_edge_df.copy()

    l_addr_edge_df = l_addr_edge_df.drop(['TFIDR', 'ARIDR', 'RFROMHN',
                                           'RTOHN', 'ZIPR', 'PARITYR',
                                           'PLUS4R', 'RFROMTYP', 'RTOTYP',
                                           'OFFSETR', 'RFROMADD', 'RTOADD'],
                                         axis=1)
    r_addr_edge_df = r_addr_edge_df.drop(['TFIDL', 'ARIDL', 'LFROMHN', 
                                          'LTOHN', 'ZIPL', 'PARITYL',
                                          'PLUS4L', 'LFROMTYP', 'LTOTYP',
                                          'OFFSETL', 'LFROMADD', 'LTOADD'],
                                         axis=1)

    l_addr_edge_df["SIDE"] = "L"
    r_addr_edge_df["SIDE"] = "R"

    l_addr_edge_df = l_addr_edge_df.rename(
        columns={"TFIDL": "TFID",
                 "ARIDL": "ARID",
                 "LFROMHN": "FROMHN",
                 "LTOHN": "TOHN",
                 "ZIPL": "ZIP",
                 "PARITYL": "PARITY",
                 "PLUS4L": "PLUS4",
                 "LFROMTYP":"FROMTYP",
                 "LTOTYP": "TOTYP",
                 "OFFSETL": "OFFSET",
                 "LFROMADD": "FROMADD",
                 "LTOADD": "TOADD" }
    )
    r_addr_edge_df = r_addr_edge_df.rename(
        columns={"TFIDR": "TFID",
                 "ARIDR": "ARID",
                 "RFROMHN": "FROMHN",
                 "RTOHN": "TOHN",
                 "ZIPR": "ZIP",
                 "PARITYR": "PARITY",
                 "PLUS4R": "PLUS4",
                 "RFROMTYP":"FROMTYP",
                 "RTOTYP": "TOTYP",
                 "OFFSETR": "OFFSET",
                 "RFROMADD": "FROMADD",
                 "RTOADD": "TOADD"}
    )
    return (l_addr_edge_df, r_addr_edge_df)


def concat_shapes(df_list):
    """Concatenate lists of shapefile data"""
    df = pd.concat(df_list, axis=0)
    return (df)


def merge_shapes(df_1, df_2, merge_cols, merge_method):
    """Merge shapefile dataframes"""
    df = df_1.merge(df_2, on=merge_cols, how=merge_method)
    return (df)


class GeoLookUpBuilder():
    """
    This class builds geo lookup tables.
    
    Parameters
    ----------
    support_files_path: str
        Path to support files
    year: str (default 2019)
        Year associated with ACS data
    output_folder_suffix: str
        Suffix attached to the output folder name.
    """

    def __init__(self, support_files_path, year, output_folder_suffix):
        self.support_files_path = support_files_path
        self.year = year
        self.output_folder_suffix = output_folder_suffix
        self.raw_geo_path = os.path.join(self.support_files_path, "raw/geo", self.year)
        self.out_geo_path = os.path.join(self.support_files_path, "processed/geo", f"{self.year}_{self.output_folder_suffix}")

    def fit(self):
        return self

    def transform(self, st_cty_code, save_table=True):
        """
        Returns DataFrame of geo lookup tables

        Parameters
        ----------
        st_cty_code: str
            state county code string
        save_table: bool
            Indicator to save dataframe
        """
        curpath = dirname(__file__)
        make_directory(output_directory=self.raw_geo_path)
        make_directory(output_directory=self.out_geo_path)

        geo_support_files_path = self.raw_geo_path
        print("Shapefile input:", geo_support_files_path)
        print("Lookup Table output:", self.out_geo_path)
        print("")

        addrfeat = "".join([geo_support_files_path,
                            "/tl_",
                            self.year,
                            "_",
                            st_cty_code,
                            "_addrfeat.shp"])
        faces = "".join([geo_support_files_path,
                         "/tl_",
                         self.year,
                         "_",
                         st_cty_code,
                         "_faces.shp"])
        edges = "".join([geo_support_files_path,
                         "/tl_",
                         self.year,
                         "_",
                         st_cty_code,
                         "_edges.shp"])

        print(" ... Loading requirements")
        af_df = gdbToDf_short(addrfeat,
                              None)
        fc_df = gdbToDf_short(faces,
                              "face")
        ed_df = gdbToDf_short(edges,
                              "edge")

        af_col_keep = [
            "TLID",
            "TFIDL",
            "TFIDR",
            "ARIDL",
            "ARIDR",
            "LINEARID",
            "FULLNAME",
            "LFROMHN",
            "LTOHN",
            "RFROMHN",
            "RTOHN",
            "ZIPL",
            "ZIPR",
            "EDGE_MTFCC",
            "ROAD_MTFCC",
            "PARITYL",
            "PARITYR",
            "LFROMTYP",
            "LTOTYP",
            "RFROMTYP",
            "RTOTYP",
            "OFFSETL",
            "OFFSETR",
            "PLUS4R",
            "PLUS4L"
        ]

        if "STATEFP20" in fc_df.columns:
            fc_df = fc_df.rename(columns={"STATEFP20": "STATEFP",
                                          "COUNTYFP20": "COUNTYFP",
                                          "TRACTCE20": "TRACTCE",
                                          "BLKGRPCE20": "BLKGRPCE",
                                          "BLOCKCE20": "BLOCKCE",
                                          "TTRACTCE20": "TTRACTCE",
                                          "TBLKGPCE20": "TBLKGPCE",
                                          "ZCTA5CE20": "ZCTA5CE",
                                          "PUMA5CE20": "PUMACE",
                                          "PUMA5CE10": "PUMACE10"
                                          }
                                 )
        else:
            fc_df[["STATEFP", "COUNTYFP", "TRACTCE", "BLKGRPCE", "BLOCKCE", "ZCTA5CE", "PUMACE"]] = fc_df[
                ["STATEFP10", "COUNTYFP10", "TRACTCE10", "BLKGRPCE10", "BLOCKCE10", "ZCTA5CE10", "PUMACE10"]]

        fc_col_keep = [
            "TFID",
            "STATEFP10",
            "COUNTYFP10",
            "TRACTCE10",
            "BLKGRPCE10",
            "BLOCKCE10",
            "ZCTA5CE10",
            "PUMACE10",
            "STATEFP",
            "COUNTYFP",
            "TRACTCE",
            "BLKGRPCE",
            "BLOCKCE",
            "ZCTA5CE",
            "TTRACTCE",
            "TBLKGPCE",
            "OFFSET",
            "PUMACE"
        ]

        ed_col_keep = [
            "STATEFP",
            "COUNTYFP",
            "TLID",
            "TFIDL",
            "TFIDR",
            "FULLNAME",
            "LFROMADD",
            "LTOADD",
            "RFROMADD",
            "RTOADD",
            "ZIPL",
            "ZIPR",
            "OFFSETL",
            "OFFSETR",
        ]

        af_df = af_df[af_col_keep]
        fc_df = fc_df[fc_col_keep]
        ed_df = ed_df[ed_col_keep]

        print(" ... Creating lookup table")

        addr_edge_df = merge_shapes(af_df,
                                    ed_df,
                                    ["TLID", "TFIDL", "TFIDR",
                                     "ZIPL", "ZIPR", "FULLNAME",
                                     "OFFSETL", "OFFSETR"],
                                    "inner")
        
        l_addr_edge_df, r_addr_edge_df = sides_split(addr_edge_df)

        l_addr_edge_face_df = merge_shapes(l_addr_edge_df,
                                           fc_df,
                                           ["TFID", "COUNTYFP", "STATEFP", "OFFSET"],
                                           "inner")
        r_addr_edge_face_df = merge_shapes(r_addr_edge_df,
                                           fc_df,
                                           ["TFID", "COUNTYFP", "STATEFP", "OFFSET"],
                                           "inner")
        
        l_addr_edge_face_df = l_addr_edge_face_df[l_addr_edge_face_df["ZIP"].notna()] 
        r_addr_edge_face_df = r_addr_edge_face_df[r_addr_edge_face_df["ZIP"].notna()]         

        aef = concat_shapes([l_addr_edge_face_df,
                             r_addr_edge_face_df])

        aef.drop_duplicates(inplace=True)

        print(" ... Formatting lookup table")
        # rename features to avoid user overlap
        aef = aef.rename(columns={"ZIP": "ZEST_ZIP",
                                  "FULLNAME": "ZEST_FULLNAME"})

        aef["RAW_ZEST_ZIP"] = aef["ZEST_ZIP"].copy()
        aef["RAW_ZEST_STATEFP"] = aef["STATEFP"].copy()
        aef["RAW_ZEST_COUNTYFP"] = aef["COUNTYFP"].copy()
        aef["RAW_ZEST_FULLNAME"] = aef["ZEST_FULLNAME"].copy()
        aef["RAW_ZEST_TRACTCE"] = aef["TRACTCE"].copy()
        aef["RAW_ZEST_BLKGRPCE"] = aef["BLKGRPCE"].copy()

        aef = aef.astype(str)
        gps = ProcessGLookUp()

        data_path = join(curpath, '../data/processed/state_mapping.json')
        state_mapping = load_json(data_path)

        aef = gps.transform(aef, state_mapping)    
        aef = aef.sort_values(['PARITY', 'ZEST_ZIP', 'ZEST_FULLNAME', 'FROMHN_LEFT', 'FROMHN_RIGHT'])                    
        aef = aef.reset_index(drop = True)
        
        final_cols = ['STATEFP', 'COUNTYFP', 'TRACTCE', 'BLKGRPCE', 'ZEST_FULLNAME', 'FROMHN' , 'TOHN', 'ZEST_ZIP', 
                      'ZCTA5CE', 'ZCTA5CE10', 'FROMHN_LEFT', 'FROMHN_RIGHT', 'TOHN_LEFT', 'TOHN_RIGHT', 'PARITY']
        aef = aef[final_cols]                    
        # optional save 
        if save_table:
            make_directory(self.out_geo_path)
            filename = "".join(["Zest_Geo_Lookup_", self.year, "_", st_cty_code, ".parquet"])
            print(f"Saving {filename}")
            save_dataframe(aef, self.out_geo_path, filename)
        else:
            print("No tables were saved")

        return (aef)

class GeoLookUpLooper(GeoLookUpBuilder):
    """
    This class loops through GeoLookUpBuilder() to build geo lookup tables
    
    Parameters
    ----------
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self):
        """
        Collects a dictionary of files to loop through

        Parameters
        ----------
        """          
        self.st_dict = dict()
        shp_list = [shp_file for shp_file in os.listdir(self.raw_geo_path) if ('_edges.shp' in shp_file
                                                                                or '_faces.shp' in shp_file
                                                                                or '_addrfeat.shp' in shp_file
                                                                                    )
                                                                                and '.iso.xml' not in shp_file]
        strings_count = pd.Series(shp_list).str[8:13].value_counts()
        st_cty_code_list = list(strings_count[strings_count == 3].sort_index().index)

        st_code_list = [st[0:2] for st in st_cty_code_list]
        st_code_list = np.unique(np.array(st_code_list))

        for st_code in st_code_list:
            self.st_dict[st_code] = [st_cty for st_cty in st_cty_code_list if st_cty[:2] == st_code]
        return self

    def __st_cty_builder(self, st_cty_code, geo_build, save_table):
        """
        Builds one lookup table on st_cty_code level

        Parameters
        ----------
        """       
        try:
            geo_build.transform(st_cty_code, save_table = save_table)
        except: 
            pass
    
    def transform(self, st_list = None, save_st_cty_tables=True, save_st_tables=True, n_jobs = -1):
        """
        Returns DataFrame of geo lookup tables

        Parameters
        ----------
        st_list: list
            list of state codes to process
        save_st_cty_tables: bool
            Indicator to save state-county level tables
        save_st_tables: bool
            Indicator to save state level tables
        n_jobs: int (default -1)
            Number of jobs in parallel
        """        

        if st_list is None:
            temp_st_dict = self.st_dict
        else:
            temp_st_dict = {key: self.st_dict[key] for key in st_list}
        
        if save_st_cty_tables:
            geo_build = GeoLookUpBuilder(support_files_path = self.support_files_path, 
                                     year = self.year, 
                                     output_folder_suffix = self.output_folder_suffix)
            for st_code in temp_st_dict:
                print(f"Working on st_cty tables (st_code = {st_code})")
                Parallel(n_jobs = n_jobs, verbose=1)(delayed(self.__st_cty_builder)(st_cty_code, geo_build, save_st_cty_tables) for st_cty_code in temp_st_dict[st_code])

        if save_st_tables:
            for st_code in temp_st_dict:
                print(f"Working on st tables (st_code = {st_code})")
                output_list = []
                for st_cty_code in temp_st_dict[st_code]:
                    try:
                        input_filename = "".join(["Zest_Geo_Lookup_", self.year, "_", st_cty_code, ".parquet"])
                        input_filepath = os.path.join(self.out_geo_path, input_filename)
                        output_list.append(pd.read_parquet(input_filepath))
                    except:
                        pass
                output = pd.concat(output_list).reset_index(drop = True)
                output_filename = "".join(["Zest_Geo_Lookup_", self.year, "_State_", st_code, ".parquet"])
                save_dataframe(output, self.out_geo_path, output_filename)

        return (self)
