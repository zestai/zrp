Preparing A New Release
_______________________

Context
=======
There are two facests to providing new ZRP releases. This is because the ZRP requires additional files to function that aren't available when you git clone or pip install. This is due to the fact that the pre-build pipelines and lookup tables that the ZRP requires are rather large and aren't able to be stored conventionally on github. We, thus, utilize Github's Releases functionality to bundle the souce code as a release, and with it, attach two zip files: 1) the lookup tables, and 2) the pipelines. 

The additional lookup tables and pipelines are installed after you pip install or clone the package by running the zrp modeule, download.py. This module downloads the two zips, unzips them, files the contents, and then removes the unnecessary downloaded folders. Ultimately, after running this module (steps shown in the README), the acs and geo lookup tables will be stored with the following paths:
::

  zrp/data/processed/acs/{acs_year}/{acs_range}/*.parquet

  Ex.: zrp/zrp/data/processed/acs/2019/5yr/processed_Zest_ACS_Lookup_20195yr_blockgroup.parquet
  
  zrp/data/processed/geo/{geo_year}/*.parquet

  Ex.: zrp/zrp/data/processed/geo/2019/Zest_Geo_Lookup_2019_State_01.parquet

The pipelines will be stored with the following paths:
::

  zrp/modeling/models/{geo_level}/pipe.pkl  (where geo_level might be 'block_group', 'census_tract', or 'zip_code')
  
  Ex.: zrp/modeling/models/block_group/pipe.pkl


Preparing the release
=====================

Observe the following steps in order to safely and correctly prepare and push new Pypi, and Github releases for the zrp packages.

If you contribute to the ZRP tool via patches, feature upgrades, or code overhauls that **DO NOT** affect the lookup tables or pipelines in use:

#. Push all changes to the Github
#. Follow the steps below to prepare a new Pypi package
    * **NOTE:** when you follow step 2, similar to the example provided, you are to bump the version in setup.py
  
If, however, you generate new lookup tables, or pipelines, in your contribution: 

#. Push all changes to the Github. 
    * Ensure that the acs/, geo/ data folders, and the {geo_level}/pipe.pkl files are not tracked in git and are added to the gitignore. 

#. Follow the steps below to prepare a new Pypi package
    * **NOTE:** when you follow step 2, and bump the version, not only should you bump the version in setup.py, but you must also open up zrp/about.py and set the version there to the now updated (bumped) version in setup.py

#. Follow the steps below to prepare a new Github Release that includes the lookup tables and pipelines zips.


Pypi 
====
(`Reference <https://widdowquinn.github.io/coding/update-pypi-package/>`_)

#. Once you've updated the package source code, ensure you have an up to date local repo and push/merge all commits to Github

#. Incremenet the version number for the package. We use the tool `Bump Version <https://pypi.org/project/bumpversion/>`_ to ensure all version numbers are kept consistent. You can install Bumpversion from PyPI:
    ::

      $ pip install bumpversion

    Bumpversion is used as follows:
    ::

    $ bumpversion --current-version ?.?.? [major/minor/patch] [<file_names>]


    For example, to increment the MINOR version of zrp, you would do something like this:
    ::

    $ bumpversion --current-version 0.1.0 minor setup.py

#. Update local packages for distribution
    ::

      python -m pip install --user --upgrade setuptools wheel
      python -m pip install --user --upgrade twine

#. Create distribution packages on your local machine, and check the dist/ directory for the new version files
    ::

      python setup.py sdist bdist_wheel
      ls dist

#. Remove the old package version distribution packages in '/dist'


#. Upload the distribution files to pypi’s test server
    ::

      python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

    * Check the upload on the test.pypi server [https://test.pypi.org/project/PACKAGE/VERSION/]
  
#. Test the upload with a local installation
    ::

      python -m pip install --index-url https://test.pypi.org/simple/ --no-deps <PACKAGE>
  
    * Start Python, import the package, and test the version

#. Upload the distribution files to pypi
    ::

      python -m twine upload dist/*
  
    * Check the upload at pypi [https://pypi.org/project///](https://pypi.org/project///)


Github Releases
===============

#. On the repo home Github page, in the right hand column, click "Releases"

#. Click "Draft a new release"

#. Enter the new relesae title in the following format: "zrp" + "-" + "VERSION"

    * Ex.: "zrp-0.1.0"

    * Ensure that the VERSION is the same version as the pypi deployment you just created

#. Select "Choose a tag", and generate a new tag with the same name as the title selected in step 3

#. Enter in any details describing the release

#. Click "Attach binaries by dropping them here or selecting them" and select the pipelines.zip and lookup_tables.zip zips you've generated

    * pipelines.zip, when unzipped, should be a folder with the following structure:
::

| pipelines
| |
| |----- block_group_pipe.pkl
| |----- census_tract_pipe.pkl
| |----- zip_code_pipe.pkl
 
    * lookup_tables.zip, when unzipped, should be a folder with the following structure ('2019' and '5yr' may be replaced by whatever ACS year and year range is applicable for the acs lookup table data you're uploading):
::

| ├── lookup_tables
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
    
    

5. Publish Release



