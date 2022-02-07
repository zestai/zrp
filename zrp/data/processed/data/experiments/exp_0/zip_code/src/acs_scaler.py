import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin 


class CustomRatios(BaseEstimator, TransformerMixin):
        """
    CustomRatios is used to convert the counts into percentages and ratios for ACS data, where possible. 
    """
        def __init__(self):
            self.house_val_cols = ['B25075_002',
                                    'B25075_003',
                                    'B25075_004',
                                    'B25075_005',
                                    'B25075_006',
                                    'B25075_007',
                                    'B25075_008',
                                    'B25075_009',
                                    'B25075_010',
                                    'B25075_011',
                                    'B25075_012',
                                    'B25075_013',
                                    'B25075_014',
                                    'B25075_015',
                                    'B25075_016',
                                    'B25075_017',
                                    'B25075_018',
                                    'B25075_019',
                                    'B25075_020',
                                    'B25075_021',
                                    'B25075_022',
                                    'B25075_023',
                                    'B25075_024',
                                    'B25075_025',
                                    'B25075_026',
                                    'B25075_027'
                                  ]
            self.income_cols = [    'B19001_002',
                                    'B19001_003',
                                    'B19001_004',
                                    'B19001_005',
                                    'B19001_006',
                                    'B19001_007',
                                    'B19001_008',
                                    'B19001_009',
                                    'B19001_010',
                                    'B19001_011',
                                    'B19001_012',
                                    'B19001_013',
                                    'B19001_014',
                                    'B19001_015',
                                    'B19001_016',
                                    'B19001_017']
            self.education_cols = ['B06009_001',
                                   'B06009_002',
                                        'B06009_003',
                                        'B06009_004',
                                        'B06009_005',
                                        'B06009_006']
            self.race_cols = ['B02001_001',
                              'B02001_002', 
                                'B02001_003', 
                                'B02001_004', 
                                'B02001_005', 
                                'B02001_006', 
                                'B02001_007', 
                                'B02001_008', 
                                'B02001_009', 
                                'B02001_010']
            self.transit_cols = [    'B08301_002',
                                    'B08301_003',
                                    'B08301_004',
                                    'B08301_011', 
                                    'B08301_012', 
                                    'B08301_013', 
                                    'B08301_016', 
                                    'B08301_017', 
                                    'B08301_018',
                                    'B08301_019',
                                    'B08301_020', 
                                    'B08301_021']
            self.hispanic_cols = ["B03001_001",
                                  "B03001_002", 
                                  "B03001_003",
                                 "B03001_008",
                                  "B03001_016"]
            self.single_ancestry_cols = ["B04004_001", 
                                  "B04004_006", 
                                  "B04004_035", 
                                  "B04004_038", 
                                  "B04004_073",
                                  "B04004_094"]
            self.reporting_ancestry_cols = ["B04006_001", 
                                  "B04006_006", 
                                  "B04006_035", 
                                  "B04006_038", 
                                  "B04006_073",
                                  "B04006_094"]
            self.ancestry_cols = ["B04007_001",
                                 "B04007_002",
                                 "B04007_005"]
            self.nativity_cols = ["B05012_001",
                                  "B05012_002",
                                  "B05012_003"]
            self.language_cols = ["C16001_001",
                                  "C16001_002",
                                  "C16001_003",
                                  "C16001_006",
                                  "C16001_009",
                                  "C16001_012",  
                                  "C16001_015", 
                                  "C16001_018",  
                                  "C16001_021",
                                  "C16001_024",  
                                  "C16001_029",  
                                  "C16001_030", 
                                  "C16001_033",  
                                  "C16001_036"]
            self.naturalization_cols = ["B05011_001",
                                        "B05011_002",
                                        "B05011_003",
                                        "B05011_004",
                                        "B05011_005", 
                                        "B05011_006",   
                                        "B05011_007", 
                                        "B05011_008",  
                                        "B05011_009",
                                        "B05011_010"]
            
        def fit(self, X, y):
            self.data_cols = list(X.columns)
            self.house_val_cols = list(set(self.house_val_cols).intersection(set(self.data_cols)))
            self.income_cols = list(set(self.income_cols).intersection(set(self.data_cols)))
            self.education_cols = list(set(self.education_cols).intersection(set(self.data_cols)))
            self.race_cols = list(set(self.race_cols).intersection(set(self.data_cols)))
            self.transit_cols = list(set(self.transit_cols).intersection(set(self.data_cols)))
            self.hispanic_cols = list(set(self.hispanic_cols).intersection(set(self.data_cols)))
            self.single_ancestry_cols = list(set(self.single_ancestry_cols).intersection(set(self.data_cols)))
            self.reporting_ancestry_cols = list(set(self.reporting_ancestry_cols).intersection(set(self.data_cols)))
            self.ancestry_cols = list(set(self.ancestry_cols).intersection(set(self.data_cols)))
            self.nativity_cols = list(set(self.nativity_cols).intersection(set(self.data_cols)))
            self.language_cols = list(set(self.language_cols).intersection(set(self.data_cols)))
            self.naturalization_cols = list(set(self.naturalization_cols).intersection(set(self.data_cols)))
            return self
 
                                  
        def transform(self, X):
            data= X.copy()
            
            if 'B99021_001' in self.data_cols:
                data['allocated_race'] = round(data['B99021_002'] / data ['B99021_001'], 6)
                data['unallocated_race'] = round(data['B99021_003'] / data ['B99021_001'], 6)

            # work 
            if ('B23020_001' in self.data_cols)  & ('B23020_003' in self.data_cols)  & ('B23020_002' in self.data_cols):
                data['weekly_work_ratio'] =  data['B23020_001']/40
                data['weekly_work_ratio_m'] =  data['B23020_002']/40
                data['weekly_work_ratio_w'] =  data['B23020_003']/40
                data['weekly_work_ratio_wm'] = round(data['B23020_003']/data['B23020_002'], 2)
                data['weekly_work_sum'] = data['B23020_003'] + data['B23020_002']

            # house value 
            for col in self.house_val_cols:
                data[col+'_pct'] = round(data[col]/ data['B01003_001'], 5)
                
            # income
            for col in self.income_cols:
                data[col+'_pct'] = round(data[col]/ data['B01003_001'], 4)
                
            # education 
            for col in self.education_cols:
                data[col+'_pct'] = round(data[col]/ data['B01003_001'], 4)
            # hispanic_cols  
            for col in self.hispanic_cols:
                data[col+'_pct'] = round(data[col]/ data['B01003_001'], 5)
                
            # single_ancestry_cols
            for col in self.single_ancestry_cols:
                data[col+'_pct'] = round(data[col]/ data['B01003_001'], 5)
                
            # reporting_ancestry_cols 
            for col in self.reporting_ancestry_cols:
                data[col+'_pct'] = round(data[col]/ data['B01003_001'], 5)
            # ancestry_cols  
            for col in self.ancestry_cols:
                data[col+'_pct'] = round(data[col]/ data['B01003_001'], 4)
                
            # nativity_cols
            for col in self.nativity_cols:
                data[col+'_pct'] = round(data[col]/ data['B01003_001'], 5)
                
            # language_cols 
            for col in self.language_cols:
                data[col+'_pct'] = round(data[col]/ data['B01003_001'], 5)                
            # naturalization 
            for col in self.naturalization_cols:
                data[col+'_pct'] = round(data[col]/ data['B01003_001'], 5)    

            # race 
            for col in self.race_cols:
                data[col+'_pct'] = round(data[col]/ data['B01003_001'], 5) 
                
            data['race_minority_pct'] =  round((data['B02001_003']+ data['B02001_004']+ data['B02001_005']+ data['B02001_006']+ data['B02001_008']) / data['B02001_001'])
            
            data['race_minority_red_pct'] =  round((data['B02001_003']+ data['B02001_004']+ data['B02001_006']) / data['B02001_001'], 6)
            
            data['race_min'] =  round(data['B02001_003']+ data['B02001_004']+ data['B02001_006'], 6)
            
            data['race_minority_allocation_pct'] =  round((data['B02001_003']+ data['B02001_004']+ data['B02001_005']+ data['B02001_006']+ data['B02001_008'])/ data['B99021_001'], 6)
            data['race_aaw'] = round(data['B02001_005']/data['B02001_002'], 6)
            
            data['race_minw'] = round((data['B02001_003']+ data['B02001_004']+ data['B02001_005']+ data['B02001_006']+ data['B02001_008'])/data['B02001_002'], 4)
            
            data['race_aian'] = round(data['B02001_004']/ data["B02001_001"], 6)
            # add language, ethinicity, origin
            if ('B03001_003' in self.data_cols) & ('B03001_001' in self.data_cols):
                data['ethnicity_hispanic'] = round(data['B03001_003']/data['B03001_001'], 6)
            if 'B05012_003' in self.data_cols:
                data['pct_multinational'] = round(data['B05012_003']/data['B05012_001'], 4)
            if 'C16001_030' in self.data_cols:
                data['pct_aapi_language'] = round((data["C16001_018"] + data["C16001_021"] + data["C16001_024"] + data["C16001_029"] + data["C16001_030"]) / data["C16001_001"], 6)
            if 'C16001_003' in self.data_cols:
                data['pct_spanish'] = round(data["C16001_003"] / data["C16001_001"], 4)
            # transit
            for col in self.transit_cols:
                data[col+'_pct'] = round(data[col]/ data['B01003_001'],5) 
                
            raw_counts = list(self.house_val_cols + self.income_cols +  self.education_cols +  self.transit_cols  +  self.single_ancestry_cols +  self.ancestry_cols +  self.nativity_cols +  self.naturalization_cols + self.reporting_ancestry_cols + self.language_cols)
            raw_counts_list = list(set(data.columns).intersection(set(raw_counts)))
            data = data.drop(raw_counts_list, axis=1)
            data = data.replace([np.inf, -np.inf], np.nan)
            return(data)
                

            
