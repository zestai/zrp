import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.rst").read_text()

# The contents of the requirements.txt
with open("requirements.txt", 'r') as file:
    requirements_raw = file.readlines()

requirements = []
for r in requirements_raw:
    requirements.append(r.replace('\n', ''))

# This call to setup() does all the work
setup(
    name="zrp",
    version="0.1.2",
    description="The Zest Race Predictor tool predicts race/ethnicity using a name and address as inputs.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/zestai/zrp",
    author="Kasey Matthews et al.",
    author_email="abetterway@zest.ai",
    license="Apache-2.0",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    keywords="race ethnicity names address acs geocode",
    packages=find_packages(exclude=("extras")),
    package_data={  # TODO
        "zrp": [
            # "data/processed/acs/2019/5yr/processed_Zest_ACS_Lookup_20195yr_blockgroup.parquet",
            # "data/processed/acs/2019/5yr/processed_Zest_ACS_Lookup_20195yr_tract.parquet",
            # "data/processed/acs/2019/5yr/processed_Zest_ACS_Lookup_20195yr_zip.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_11.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_10.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_15.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_44.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_09.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_33.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_25.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_04.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_50.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_23.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_32.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_34.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_56.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_24.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_02.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_49.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_35.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_41.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_53.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_16.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_45.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_38.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_30.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_54.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_46.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_36.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_06.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_08.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_01.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_22.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_72.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_05.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_12.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_55.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_42.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_40.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_28.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_27.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_26.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_31.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_19.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_39.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_18.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_47.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_37.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_20.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_17.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_29.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_21.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_51.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_13.parquet",
            # "data/processed/geo/2019/Zest_Geo_Lookup_2019_State_48.parquet",
        ],
    },
    include_package_data=True,
    install_requires=[],  # TODO: put back requirements variable
    entry_points={
        "console_scripts": [

            # TODO: Once we know what endpoints are being exposed (see ethnicolr's as example)
        ]
    },
)  # TODO: Update Manifest.in file to include data and other files as suggested here: https://realpython.com/pypi-publish-python-package/#adding-files-to-your-package
