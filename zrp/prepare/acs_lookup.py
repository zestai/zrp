from .preprocessing import *
from os.path import dirname, join, expanduser
from joblib import Parallel, delayed
from .utils import *
from tqdm import tqdm
import pandas as pd
import numpy as np
import censusdata
import json
import sys
import os
import re


def pad_logrecno(data):
    data["LOGRECNO"] = data["LOGRECNO"].str.zfill(7)
    return (data)


class ACS_Parser():
    """
    Parses the raw ACS data
    
    Parameters:
    ----------
    support_files_path: str
        Support files path pointing to where the raw ACS data is stored
    year: str (default 2019)
        Year of ACS data. 
    span: str (default 5)
        Span of ACS data. The ACS data is available in 1 or 5 year spans. The 5yr ACS data is the most comprehensive & is available at more granular levels than 1yr data.
    state_level: str
        State to parse
    n_jobs: int
        Number of jobs in parallel        
    """
    
    def __init__(self, support_files_path, state_level, n_jobs=-1, year='2019', span='5'):
        self.support_files_path = support_files_path
        self.year = year
        self.span = span
        self.state_level = state_level
        self.n_jobs = n_jobs
        self.raw_acs_path = os.path.join(self.support_files_path,
                                         f"raw/acs/{self.year}/{self.span}yr")
        self.out_acs_path = os.path.join(self.support_files_path,
                                         "parsed/acs",
                                         self.year,
                                         f"{self.span}yr")
        self.fol_acs_tmp = f"{self.year}_{self.span}yr_Summary_FileTemplates/"

    def fit(self):
        return self

    def acs_geo(self, data):
        """
        Merges input data with ACS data
        
        Parameters:
        ----------
        data: dataframe
            Original input dataframe
        """
        geo_file_path = os.path.join(self.raw_acs_path,
                                     self.fol_acs_tmp,
                                     f'{self.span}_year_Mini_Geo.xlsx')

        ag_df = pd.read_excel(geo_file_path,
                              sheet_name=self.state_level,
                              dtype=str)
        ag_df = ag_df.rename(columns={'Logical Record Number': 'LOGRECNO',
                                      'Geography ID': 'GEOID'})

        data = data.merge(ag_df,
                          on=["LOGRECNO"])
        return (data)

    def acs_track_5yr(self):
        """
        Initializes 5 year sequence dictionary
            
        """
        file_path_list = []
        sequence_dict = {}
        for root, dirs, files in os.walk(os.path.join(self.raw_acs_path,
                                                      "Tracts_Block_Groups")):
            for file in files:
                if ('000.txt' in file) & \
                        ('e' in file) & \
                        (f'{self.span}{self.state_level}' in file):
                    file_path_list.append("".join([root, file]))
                    sequence = file.split('000.txt')[0].split(self.state_level)[1]
                    sequence = str(int(sequence))
                    sequence_dict[sequence] = {}

                    sequence_dict[sequence]["data"] = None
                    sequence_dict[sequence]["sequence"] = None
                    sequence_dict[sequence]["headers"] = None

                    sequence_dict[sequence]['Tracts_Block_Groups'] = {}
                    sequence_dict[sequence]['Not_Tracts_Block_Groups'] = {}

                    sequence_dict[sequence]['Tracts_Block_Groups']["file"] = file
                    sequence_dict[sequence]['Not_Tracts_Block_Groups']["file"] = None

                    sequence_dict[sequence]['Tracts_Block_Groups']["data"] = None
                    sequence_dict[sequence]['Not_Tracts_Block_Groups']["data"] = None

        for root, dirs, files in os.walk(os.path.join(self.raw_acs_path, "Not_Tracts_Block_Groups")):
            for file in files:
                if ('000.txt' in file) & ('e' in file) & \
                        (f'{self.span}{self.state_level}' in file):
                    file_path_list.append("".join([root, file]))
                    sequence = file.split('000.txt')[0].split(self.state_level)[1]
                    sequence = str(int(sequence))
                    sequence_dict[sequence]['Not_Tracts_Block_Groups']["file"] = file
        return (sequence_dict)

    def acs_track_1yr(self):
        """
        Initializes 1 year sequence dictionary
            
        """
        file_path_list = []
        sequence_dict = {}
        for root, dirs, files in os.walk(os.path.join(self.raw_acs_path,
                                                      "All_Geographies")):
            for file in files:
                if ('000.txt' in file) & \
                        ('e' in file) & \
                        (f'{self.span}{self.state_level}' in file):
                    file_path_list.append("".join([root, file]))
                    sequence = file.split('000.txt')[0].split(self.state_level)[1]
                    sequence = str(int(sequence))
                    sequence_dict[sequence] = {}
                    sequence_dict[sequence]["file"] = file
                    sequence_dict[sequence]["sequence"] = None
                    sequence_dict[sequence]["data"] = None
                    sequence_dict[sequence]["headers"] = None

        return (sequence_dict)

    def acs_parse_5yr(self, sequence_dict, i, save_table):
        """
        Parses 5 year ACS data
        
        Parameters:
        ----------
        sequence_dict: dict
            State level dictionary that contains file paths & sequence names
        i: int
            Sequence iteration
        save_table: bool
            Indicating is table will be saved to file
            
        """
        print("... Generating ACS table", self.state_level, "sequence", i)

        dat_file_1 = os.path.join(self.raw_acs_path,
                                  "Tracts_Block_Groups",
                                  sequence_dict[str(i)]['Tracts_Block_Groups']['file'])
        dat_file_2 = os.path.join(self.raw_acs_path,
                                  "Not_Tracts_Block_Groups",
                                  sequence_dict[str(i)]['Not_Tracts_Block_Groups']['file'])

        try:
            tmp_data_1 = pd.read_csv(dat_file_1, sep=",", header=None, dtype=str)
            tmp_data_2 = pd.read_csv(dat_file_2, sep=",", header=None, dtype=str)

            seq_file = os.path.join(self.raw_acs_path,
                                    sequence_dict[str(i)]['sequence'])

            tmp_headers = pd.read_excel(seq_file, sheet_name='e', dtype=str)
            sequence_dict[str(i)]["headers"] = tmp_headers

            feature_mapping = sequence_dict[str(i)]['headers'].to_dict('records')
            sequence_dict[str(i)]['description'] = feature_mapping[0]

            new_col_names = list(tmp_headers.columns)
            tmp_data_1.columns = new_col_names
            tmp_data_2.columns = new_col_names

        except pd.errors.EmptyDataError:
            print(f'   Note: {dat_file_1} was empty.')
            tmp_data = pd.read_csv(dat_file_2, sep=",", header=None, dtype=str)

            seq_file = os.path.join(self.raw_acs_path,
                                    sequence_dict[str(i)]['sequence'])
            tmp_headers = pd.read_excel(seq_file, sheet_name='e', dtype=str)

            sequence_dict[str(i)]["headers"] = tmp_headers

            feature_mapping = sequence_dict[str(i)]['headers'].to_dict('records')
            sequence_dict[str(i)]['description'] = feature_mapping[0]

            new_col_names = list(tmp_headers.columns)
            tmp_data.columns = new_col_names

        tmp_data = pad_logrecno(tmp_data)
        tmp_data = self.acs_geo(tmp_data)

        sequence_dict[str(i)]["data"] = tmp_data

        if save_table:
            file_name = "".join(["Zest_ACS_", self.state_level, "_seq",
                                 str(i), "_", self.year, '_', self.span,
                                 'yr.parquet'])
            save_dataframe(tmp_data, self.out_acs_path, file_name)
        return (sequence_dict[str(i)])

    def acs_parse_1yr(self, sequence_dict, i, save_table):
        """
        Parses 1 year ACS data
        
        Parameters:
        ----------
        sequence_dict: dict
            State level dictionary that contains file paths & sequence names
        i: int
            Sequence iteration
        save_table: bool
            Indicating is table will be saved to file
            
        """
        print("... Generating ACS table", self.state_level, "sequence", i)
        dat_file = os.path.join(self.raw_acs_path,
                                "All_Geographies",
                                sequence_dict[str(i)]['file'])

        tmp_data = pd.read_csv(dat_file, sep=",", header=None, dtype=str)
        seq_file = os.path.join(self.raw_acs_path,
                                sequence_dict[str(i)]['sequence'])
        tmp_headers = pd.read_excel(seq_file, sep=",",
                                    sheet_name="e", dtype=str)

        sequence_dict[str(i)]["headers"] = tmp_headers
        feature_mapping = sequence_dict[str(i)]['headers'].to_dict('records')
        sequence_dict[str(i)]['description'] = feature_mapping[0]

        new_col_names = list(tmp_headers.columns)
        tmp_data.columns = new_col_names
        tmp_data = pad_logrecno(tmp_data)
        tmp_data = self.acs_geo(tmp_data)

        sequence_dict[str(i)]["data"] = tmp_data

        if save_table:
            file_name = "".join(["Zest_ACS_", self.state_level, "_seq", str(i),
                                 "_", self.year, '_', self.span, 'yr.parquet'])

            save_dataframe(tmp_data, self.out_acs_path, file_name)
        return (sequence_dict[str(i)])

    def transform(self, save_table=True):
        """
        Parameters
        ----------
        save_table: bool
            Optional save
        """
        if save_table:

            make_directory(output_directory=self.raw_acs_path)
            make_directory(output_directory=self.out_acs_path)
            make_directory(output_directory=os.path.join(self.raw_acs_path,
                                                         self.fol_acs_tmp))
        if self.span == '5':
            sequence_dict = self.acs_track_5yr()
            for root, dirs, files in os.walk(os.path.join(self.raw_acs_path,
                                                          self.fol_acs_tmp)):
                for file in files:
                    if ('.xlsx' in file) & ('Geo' not in file):
                        sequence = file.split('seq')[1].split('.xlsx')[0]
                        sequence = str(int(sequence))
                        seq_str = "".join([self.fol_acs_tmp, "seq", sequence, ".xlsx"])
                        sequence_dict[sequence]["sequence"] = seq_str


            results = Parallel(n_jobs=self.n_jobs, verbose=1)(
                delayed((self.acs_parse_5yr))(sequence_dict, sni, save_table) for sni in
                tqdm(list(sequence_dict.keys())))

        elif self.span == '1':
            sequence_dict = self.acs_track_1yr()
            for root, dirs, files in os.walk(os.path.join(self.raw_acs_path,
                                                          self.fol_acs_tmp)):
                for file in files:
                    if ('.xlsx' in file) & ('Geo' not in file):
                        sequence = file.split('seq')[1].split('.xlsx')[0]
                        sequence = str(int(sequence))
                        seq_str = "".join([self.fol_acs_tmp, "seq", sequence, ".xlsx"])
                        sequence_dict[sequence]["sequence"] = seq_str


            results = Parallel(n_jobs=self.n_jobs, verbose=1)(
                delayed((self.acs_parse_1yr))(sequence_dict, sni, save_table) for sni in
                tqdm(list(sequence_dict.keys())))
        else:
            raise ValueError('Improper ACS span provided. The only accepted values are 1 & 5')
        results_out = {}
        for d in results:
            seqn = re.findall('[0-9]{1,3}', d['sequence'])[-1]
            results_out[seqn] = {}
            results_out[seqn].update(d)
        return (results_out)


def acs_census_data(support_files_path, level):
    """Create ACS Lookup Tables using the censusdata package.
    
    Parameters:
    -----------
    level:
        Geographic level to return ACS for, options include:
        ['block group' tract', 'zip', 'county', 'state']
    
    """
    curpath = dirname(__file__)
    data_path = join(curpath, '../data/')
    misc_args = load_json(join(data_path, 'misc_args.json'))
    states = load_json(join(data_path, 'states.json'))

    df = pd.DataFrame(columns=misc_args)
    if level == 'zip':
        temp = censusdata.download(
            "acs5",
            2019,
            censusdata.censusgeo([
                ('zip code tabulation area', '*')]),
            misc_args,
        )
        df = df.append(temp)

    else:
        pass

    for i in range(len(states)):
        print("State:", i, "\n")

        if level == 'tract':
            temp = censusdata.download(
                "acs5",
                2019,
                censusdata.censusgeo([
                    ('state', states[i]),
                    ('county', '*'),
                    ('tract', '*')
                ]),
                misc_args,
            )
            df = df.append(temp)


        elif level == 'block group':
            temp = censusdata.download(
                "acs5",
                2019,
                censusdata.censusgeo([
                    ('state', states[i]),
                    ('county', '*'),
                    ('block group', '*')
                ]),
                misc_args,
            )
            df = df.append(temp)

        else:
            temp = censusdata.download(
                "acs5",
                2019,
                censusdata.censusgeo([
                    ('state', states[i]),
                    ('county', '*')
                ]),
                misc_args,
            )
            df = df.append(temp)
    return (df)


class ACS_LookupBuilder():
    """Creates a core ACS lookup table by geo level 
    
    Parameter
    --------
    geo: str
        Geo key to identify which geographic level the ACS table will be made at. 
        Three levels are currently supported zip, tract, or block group
    year: str (default 2019)
        Year of ACS data. 
    span: str
        Span of ACS data. The ACS data is available in 1 or 5 year spans. The 5yr ACS data is the most comprehensive & is available at more granular levels than 1yr data.
    n_jobs: int
        Number of jobs in parallel
    required_tables: list
        List of ACS table names to select data to include in the Lookup table
    """


    def __init__(self, support_files_path, geo, year='2019', span='5', n_jobs=-1, required_tables=None):
        self.support_files_path = support_files_path
        self.geo = geo
        self.year = year
        self.span = span
        self.n_jobs = n_jobs
        self.support_files_path = support_files_path
        curpath = dirname(__file__)
        parsed_path = join(support_files_path, f'../data/parsed/acs/{self.year}/{self.span}yr')
        processed_path = join(curpath, f'../data/processed/acs/{self.year}/{self.span}yr')
        self.raw_acs_path = parsed_path
        self.out_acs_path = processed_path
        self.required_tables = required_tables
        if self.required_tables is None:
            self.required_tables = ['B01003', 'B02001', 'B03001', 'B04004', 'B04006',
                                    'B04007', 'B05011', 'B05012', 'B06009', 'B07101',
                                    'B08301', 'B10051', 'B11017', 'B16001', 'B19001',
                                    'B23020', 'B25004', 'B25075', 'B99021', 'B99162', 'C16001']

    def acs_select_features(self, data):
        """Selects predefined ACS tables to create a lookup table"""
        prts = "|".join(self.required_tables + ['GEO', 'Geography'])
        data = data.filter(regex=prts)
        return (data)

    def acs_join_lookup(self, data, mco):
        """Appends new geo level to feature column"""
        tbl_dict = {}
        for rn in required_names:
            data = data.filter(regex="|".join([rn, 'GEO']))
            tbl_dict[rn] = data
        return (tbl_dict)

    def parsed_acs_proc(self, file, geo_pattern):
        """Creates a dataframe of geo-specific ACS data by sequence 

        Parameter
        --------
        file: str
            Filename
        geo_pattern: str
            Pattern to identify geo specific data
        """
        drop_cols = []

        long_name = 'GEO_NAME'
        long_id = 'EXT_GEOID'

        tmp_data = load_file(os.path.join(self.raw_acs_path, file))
        tmp_data = self.acs_select_features(tmp_data)

        tmp_data = tmp_data.rename(columns={'GEOID': long_id,
                                            'Geography Name': long_name})

        tmp_data = tmp_data[tmp_data[long_name].str.upper().str.contains(geo_pattern,
                                                                         regex=True)]
        na_cols = list(tmp_data.columns[tmp_data.isnull().all()])
        tmp_data = tmp_data.drop(drop_cols + na_cols,
                                 axis=1)

        tmp_data['GEOID'] = None
        tmp_data['GEOID'] = tmp_data[long_id].apply(lambda x: x.split('US')[1]).astype(str)

        if tmp_data.shape[1] < 4:
            tmp_data = pd.DataFrame()
        else:
            tmp_data = tmp_data.set_index(['GEOID',
                                           'GEO_NAME',
                                           'EXT_GEOID']
                                          ).sort_index()
        return (tmp_data)

    def transform(self, save_table):
        """
        Parameters
        ----------
        save_table: bool
            Optional save
        """
        assert self.geo in ['zip', 'tract',
                            'block group'], "Require `geo` to be specified as 'zip', 'tract' or 'block group'"
        output = []

        if self.geo == 'zip':
            geo_pattern = '^ZCTA5'

            parsed_file_list = [file for file in os.listdir(self.raw_acs_path) if 'ACS_us_seq' in file]

            output = Parallel(n_jobs=self.n_jobs, verbose=10)(
                delayed(self.parsed_acs_proc)(file, geo_pattern) for file in parsed_file_list)

        elif self.geo == 'block group':
            geo_pattern = '^BLOCK GROUP'

            parsed_file_list = [file for file in os.listdir(self.raw_acs_path) if 'ACS_us_seq' not in file]
            results = Parallel(n_jobs=self.n_jobs, verbose=1)(
                delayed(self.parsed_acs_proc)(file, geo_pattern) for file in parsed_file_list)

            tbl_dict = {}
            for rn in self.required_tables:
                tbl_dict[rn] = {}
                tbl_dict[rn]['tbls'] = []
                tbl_dict[rn]['data'] = pd.DataFrame()

            for dfd in tqdm(range(0, len(results))):
                if not results[dfd].empty:
                    mco = most_common([x.split('_')[0] for x in results[dfd].columns])[0:6]
                    tbl_dict[mco]['tbls'].append(results[dfd])

            for rn in self.required_tables:
                print("  Table:", rn)
                try:
                    tbl_dict[rn]['data'] = pd.concat(tbl_dict[rn]['tbls'])
                    print("    ", tbl_dict[rn]['data'].shape)
                    output.append(tbl_dict[rn]['data'])
                except ValueError:
                    pass
            print("")

        elif self.geo == 'tract':
            geo_pattern = '^CENSUS TRACT'

            parsed_file_list = [file for file in os.listdir(self.raw_acs_path) if 'ACS_us_seq' not in file]
            results = Parallel(n_jobs=self.n_jobs, verbose=10)(
                delayed(self.parsed_acs_proc)(file, geo_pattern) for file in parsed_file_list)

            tbl_dict = {}
            for rn in self.required_tables:
                tbl_dict[rn] = {}
                tbl_dict[rn]['tbls'] = []
                tbl_dict[rn]['data'] = pd.DataFrame()

            for dfd in tqdm(range(0, len(results))):
                if not results[dfd].empty:
                    mco = most_common([x.split('_')[0] for x in results[dfd].columns])[0:6]
                    tbl_dict[mco]['tbls'].append(results[dfd])

            for rn in self.required_tables:
                print("  Table:", rn)
                try:
                    tbl_dict[rn]['data'] = pd.concat(tbl_dict[rn]['tbls'])
                    print("    ", tbl_dict[rn]['data'].shape)
                    output.append(tbl_dict[rn]['data'])
                except ValueError:
                    pass
            print("")

        df_out = pd.concat(output, axis=1)

        # optional save 
        if save_table:
            make_directory(output_directory = self.raw_acs_path)
            make_directory(output_directory = self.out_acs_path)
            geo_name = "".join(self.geo.split())
            filename = "".join(["Zest_ACS_Lookup_", self.year, self.span, 'yr_', geo_name, '.parquet'])
            print(f"Saving {filename}")
            save_dataframe(df_out, self.out_acs_path, filename)
        else:
            print('No tables were saved')

        return (df_out)
