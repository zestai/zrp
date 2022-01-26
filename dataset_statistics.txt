North Carolina Voter Registration Data Statistics
__________________________________________________

Dataset shape
(8082149, 71)


Column space
Index(['county_id', 'county_desc', 'voter_reg_num', 'status_cd',
       'voter_status_desc', 'reason_cd', 'voter_status_reason_desc',
       'absent_ind', 'name_prefx_cd', 'last_name', 'first_name', 'middle_name',
       'name_suffix_lbl', 'res_street_address', 'res_city_desc', 'state_cd',
       'zip_code', 'mail_addr1', 'mail_addr2', 'mail_addr3', 'mail_addr4',
       'mail_city', 'mail_state', 'mail_zipcode', 'full_phone_number',
       'race_code', 'ethnic_code', 'party_cd', 'gender_code', 'birth_age',
       'birth_state', 'drivers_lic', 'registr_dt', 'precinct_abbrv',
       'precinct_desc', 'municipality_abbrv', 'municipality_desc',
       'ward_abbrv', 'ward_desc', 'cong_dist_abbrv', 'super_court_abbrv',
       'judic_dist_abbrv', 'nc_senate_abbrv', 'nc_house_abbrv',
       'county_commiss_abbrv', 'county_commiss_desc', 'township_abbrv',
       'township_desc', 'school_dist_abbrv', 'school_dist_desc',
       'fire_dist_abbrv', 'fire_dist_desc', 'water_dist_abbrv',
       'water_dist_desc', 'sewer_dist_abbrv', 'sewer_dist_desc',
       'sanit_dist_abbrv', 'sanit_dist_desc', 'rescue_dist_abbrv',
       'rescue_dist_desc', 'munic_dist_abbrv', 'munic_dist_desc',
       'dist_1_abbrv', 'dist_1_desc', 'dist_2_abbrv', 'dist_2_desc',
       'confidential_ind', 'birth_year', 'ncid', 'vtd_abbrv', 'vtd_desc'],
      dtype='object')


Race totals
W    5341774
B    1721713
U     584121
O     213544
A     107783
I      61668
M      51543
           3
Name: race_code, dtype: int64


Race by percentage
W    0.66
B    0.21
U    0.07
O    0.03
A    0.01
I    0.01
M    0.01
     0.00
Name: race_code, dtype: float64


Ethnicity by percentage
NL    0.74
UN    0.23
HL    0.03
Name: ethnic_code, dtype: float64


Gender by percentage
F    0.51
M    0.44
U    0.06
     0.00
Name: gender_code, dtype: float64


Age ranges by percentage
(20, 30]      0.17
(50, 60]      0.16
(30, 40]      0.16
(60, 70]      0.15
(40, 50]      0.15
(70, 80]      0.10
(80, 90]      0.05
(0, 20]       0.04
(90, 1000]    0.02
Name: birth_age, dtype: float64


County total
100


Counties by percentage
WAKE           0.11
MECKLENBURG    0.11
GUILFORD       0.05
FORSYTH        0.04
DURHAM         0.03
               ... 
ALLEGHANY      0.00
JONES          0.00
GRAHAM         0.00
HYDE           0.00
TYRRELL        0.00
Name: county_desc, Length: 100, dtype: float64


Zip Code Total
859


Zip codes by percentage
28269.0    0.0079
27587.0    0.0076
28277.0    0.0074
27610.0    0.0070
28078.0    0.0067
            ...  
27404.0    0.0000
27402.0    0.0000
28026.0    0.0000
27113.0    0.0000
28471.0    0.0000
Name: zip_code, Length: 859, dtype: float64



Alamaba Voter Registration Data Statistics
__________________________________________

Dataset shape
(235581, 69)


Column space
Index(['County', 'Registrant Status', 'Name Title', 'First Name',
       'Middle Name', 'Last Name', 'Name Suffix', 'Race', 'Gender',
       'Registrant ID', 'Residential Address Number',
       'Residential Address Number Suffix', 'Residential Address Direction',
       'Residential Address Name', 'Residential Address Type',
       'Residential Address Direction Suffix', 'Residential Unit Type',
       'Residential Unit Number', 'Residential City', 'Residential State',
       'Residential Zip', 'Residential Zip Segment', 'Mailing Address 1',
       'Mailing Address 2', 'Mailing Address 3', 'Mailing Address 4',
       'Mailing City', 'Mailing State', 'Mailing Zip', 'Mailing Zip Segment',
       'Precinct Part Name', 'Precinct Part Designation', 'Precinct Name',
       'Precinct Designation', 'Age', 'Date of Registration',
       'Non-Standard Physical Address', 'County Commission', 'County School',
       'Jefferson County District', 'Municipality', 'City District',
       'Senate District', 'Congressional District', 'State School District',
       'House District', 'Phone - Area Code', 'Phone Number - Exchange',
       'Phone Number - Last Four Digits', 'Last Election Voted',
       'Last Election Party Code', 'Election 2', 'Party Code 2', 'Election 3',
       'Party Code 3', 'Election 4', 'Party Code 4', 'Election 5',
       'Party Code 5', 'Election 6', 'Party Code 6', 'Election 7',
       'Party Code 7', 'Election 8', 'Party Code 8', 'Election 9',
       'Party Code 9', 'Election 10', 'Party Code 10'],
      dtype='object')


Race totals
W     166749
B      54145
H       5649
U       4378
A       2526
O        917
AI       575
F          8
Name: Race, dtype: int64


Race by percentage
W     0.71
B     0.23
H     0.02
U     0.02
A     0.01
O     0.00
AI    0.00
F     0.00
Name: Race, dtype: float64


Gender by percentage
M    0.5
F    0.5
U    0.0
Name: Gender, dtype: float64


Age ranges by percentage
(20, 30]      0.25
(30, 40]      0.18
(0, 20]       0.16
(40, 50]      0.14
(50, 60]      0.13
(60, 70]      0.09
(70, 80]      0.04
(80, 90]      0.01
(90, 1000]    0.00
Name: Age, dtype: float64


Municipality total
0


Municipalities by percentage
Series([], Name: Municipality, dtype: float64)


Zip Code Total
1139


Zip codes by percentage
35242    0.0139
35758    0.0126
36535    0.0122
36526    0.0116
36830    0.0107
          ...  
36005    0.0000
35131    0.0000
36083    0.0000
35808    0.0000
36612    0.0000
Name: Residential Zip, Length: 1139, dtype: float64



Louisiana Voter Registration Data Statistics
_____________________________________________

Dataset shape
(2889260, 49)


Column space
Index(['Jurisdiction_Parish', 'Jurisdiction_Ward', 'Jurisdiction_Precinct',
       'Personal_NamePrefix', 'Personal_FirstName', 'Personal_MiddleName',
       'Personal_LastName', 'Personal_NameSuffix', 'Residence_HouseNumber',
       'Residence_HouseFraction', 'Residence_StreetDirection',
       'Residence_StreetName', 'Residence_ApartmentNumber', 'Residence_City',
       'Residence_State', 'Residence_ZipCode5', 'Residence_ZipCode4',
       'Mail_Address1', 'Mail_Address2', 'Mail_City', 'Mail_State',
       'Mail_ZipCode5', 'Mail_ZipCode4', 'Mail_Country',
       'Jurisdiction_District_AppealsCourt', 'Jurisdiction_District_BESE',
       'Jurisdiction_District_City', 'Jurisdiction_District_Congressional',
       'Jurisdiction_District_DistrictCourt',
       'Jurisdiction_District_JusticeOfThePeace',
       'Jurisdiction_District_PoliceJury',
       'Jurisdiction_District_PublicService',
       'Jurisdiction_District_Representative',
       'Jurisdiction_District_SchoolBoard', 'Jurisdiction_District_Senate',
       'Jurisdiction_District_SupremeCourt', 'Jurisdiction_District_TaxWard',
       'Personal_Sex', 'Personal_Race', 'Registration_PoliticalPartyCode',
       'Personal_Age', 'Registration_VoterStatus', 'Registration_Date',
       'Registration_Number', 'Personal_Phone', 'LastVoted',
       'Residence_WalkListOrder', 'Personal_NameOrder',
       'Jurisdiction_District_SpecialsList'],
      dtype='object')


Race totals
W    1824983
B     900863
O      75552
H      45072
A      29302
I      13488
Name: Personal_Race, dtype: int64


Race by percentage
W    0.63
B    0.31
O    0.03
H    0.02
A    0.01
I    0.00
Name: Personal_Race, dtype: float64


Gender by percentage
F    0.55
M    0.45
U    0.00
Name: Personal_Sex, dtype: float64


Age ranges by percentage
(30, 40]      0.18
(50, 60]      0.17
(60, 70]      0.17
(40, 50]      0.16
(20, 30]      0.15
(70, 80]      0.10
(80, 90]      0.04
(0, 20]       0.03
(90, 1000]    0.01
Name: Personal_Age, dtype: float64


City total
484


Cities by percentage
NEW ORLEANS    0.09
BATON ROUGE    0.08
SHREVEPORT     0.05
LAFAYETTE      0.03
METAIRIE       0.03
               ... 
VICK           0.00
MANCHAC        0.00
JIGGER         0.00
CLARENCE       0.00
NEGREET        0.00
Name: Residence_City, Length: 484, dtype: float64


Zip Code Total
532


Zip codes by percentage
70072    0.0125
70726    0.0113
70065    0.0103
70769    0.0102
70810    0.0100
          ...  
71414    0.0000
70139    0.0000
71460    0.0000
71014    0.0000
71451    0.0000
Name: Residence_ZipCode5, Length: 532, dtype: float64



Georgia Voter Registration Data Statistics
__________________________________________


Dataset shape
(7517881, 63)


Column space
Index(['COUNTY_CODE', 'REGISTRATION_NUMBER', 'VOTER_STATUS', 'LAST_NAME',
       'FIRST_NAME', 'MIDDLE_MAIDEN_NAME', 'NAME_SUFFIX', 'NAME_TITLE',
       'RESIDENCE_HOUSE_NUMBER', 'RESIDENCE_STREET_NAME',
       'RESIDENCE_STREET_SUFFIX', 'RESIDENCE_APT_UNIT_NBR', 'RESIDENCE_CITY',
       'RESIDENCE_ZIPCODE', 'BIRTHDATE', 'REGISTRATION_DATE', 'RACE', 'GENDER',
       'LAND_DISTRICT', 'LAND_LOT', 'STATUS_REASON', 'COUNTY_PRECINCT_ID',
       'CITY_PRECINCT_ID', 'CONGRESSIONAL_DISTRICT', 'SENATE_DISTRICT',
       'HOUSE_DISTRICT', 'JUDICIAL_DISTRICT', 'COMMISSION_DISTRICT',
       'SCHOOL_DISTRICT', 'COUNTY_DISTRICTA_NAME', 'COUNTY_DISTRICTA_VALUE',
       'COUNTY_DISTRICTB_NAME', 'COUNTY_DISTRICTB_VALUE', 'MUNICIPAL_NAME',
       'MUNICIPAL_CODE', 'WARD_CITY_COUNCIL_NAME', 'WARD_CITY_COUNCIL_CODE',
       'CITY_SCHOOL_DISTRICT_NAME', 'CITY_SCHOOL_DISTRICT_VALUE',
       'CITY_DISTA_NAME', 'CITY_DISTA_VALUE', 'CITY_DISTB_NAME',
       'CITY_DISTB_VALUE', 'CITY_DISTC_NAME', 'CITY_DISTC_VALUE',
       'CITY_DISTD_NAME', 'CITY_DISTD_VALUE', 'DATE_LAST_VOTED',
       'PARTY_LAST_VOTED', 'DATE_ADDED', 'DATE_CHANGED', 'DISTRICT_COMBO',
       'RACE_DESC', 'LAST_CONTACT_DATE', 'MAIL_HOUSE_NBR', 'MAIL_STREET_NAME',
       'MAIL_APT_UNIT_NBR', 'MAIL_CITY', 'MAIL_STATE', 'MAIL_ZIPCODE',
       'MAIL_ADDRESS_2', 'MAIL_ADDRESS_3', 'MAIL_COUNTRY'],
      dtype='object')


Race totals
WH    3942674
BH    2228912
U      692780
HP     280681
AP     199593
OT     148540
AI      24701
Name: RACE, dtype: int64


Race by percentage
WH    0.52
BH    0.30
U     0.09
HP    0.04
AP    0.03
OT    0.02
AI    0.00
Name: RACE, dtype: float64


Gender by percentage
F    0.53
M    0.47
O    0.00
Name: GENDER, dtype: float64


Age ranges by percentage
County total
159


Counties by percentage
60     0.11
67     0.08
44     0.07
33     0.07
25     0.03
       ... 
30     0.00
62     0.00
152    0.00
118    0.00
131    0.00
Name: COUNTY_CODE, Length: 159, dtype: float64


Zip Code Total
572097


Zip codes by percentage
30349.0        0.0071
30043.0        0.0070
30024.0        0.0067
30044.0        0.0065
30040.0        0.0060
                ...  
314052321.0    0.0000
314052325.0    0.0000
318246631.0    0.0000
314052329.0    0.0000
310891349.0    0.0000
Name: RESIDENCE_ZIPCODE, Length: 572097, dtype: float64
