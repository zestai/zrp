from zrp.prepare.preprocessing import *
from os.path import dirname, join, expanduser
from zrp.prepare.utils import *
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
        Year associated with ACS data. 
    """

    def __init__(self, support_files_path, year):
        self.support_files_path = support_files_path
        self.year = year
        self.raw_geo_path = os.path.join(self.support_files_path, "raw/geo", self.year)
        self.out_geo_path = os.path.join(self.support_files_path, "processed/geo", f"{self.year}_backup") #new change update 

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

        # make geo keys/ids
        aef["GEOID_ZIP"] = aef["ZEST_ZIP"]
        aef["GEOID_CT"] = aef["STATEFP"] + aef["COUNTYFP"] + aef["TRACTCE"]
        aef["GEOID_BG"] = aef["GEOID_CT"] + aef["BLKGRPCE"]

        # optional save 
        if save_table:
            make_directory(self.out_geo_path)
            filename = "".join(["Zest_Geo_Lookup_", self.year, "_", st_cty_code, ".parquet"])
            print(f"Saving {filename}")
            save_dataframe(aef, self.out_geo_path, filename)
        else:
            print("No tables were saved")

        return (aef)
