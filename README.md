Zest Race Predictor
===================

[![Documentation Status](https://readthedocs.org/projects/zrp-docs/badge/?version=latest)](https://zrp-docs.readthedocs.io/en/latest/?badge=latest)
[![image](https://badge.fury.io/py/zrp.svg)](https://badge.fury.io/py/zrp)
[![image](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zestai/zrp/HEAD)
[![image](https://img.shields.io/pypi/dm/zrp.svg?label=PyPI%20downloads)](https://pypi.org/project/zrp/)

Zest Race Predictor (ZRP) is an open-source machine learning algorithm
that estimates the race/ethnicity of an individual using only their full
name and home address as inputs. ZRP improves upon the most widely used
racial and ethnic data estimation method, Bayesian Improved Surname
Geocoding (BISG), developed by RAND Corporation in 2009.

ZRP was built using ML techniques such as gradient boosting and trained
on voter data from the southeastern U.S. It was then validated on a
national sample using adjusted tract-level American Community Survey
(ACS) data. (Model training procedures are provided.)

***Compared to BISG, ZRP correctly identified:***

*   25% more African-Americans as African-American
*   35% fewer African-Americans as non-African American
*   60% fewer Whites as non-White

ZRP can be used to analyze racial equity and outcomes in critical
spheres such as health care, financial services, criminal justice, or
anywhere there's a need to impute the race or ethnicity of a population
dataset. (Usage examples are included.) The financial services industry,
for example, has struggled for years to achieve more equitable outcomes
amid charges of discrimination in lending practices.

Zest AI began developing ZRP in 2020 to improve the accuracy of our
clients' fair lending analyses by using more data and better math. We
believe ZRP can greatly improve our understanding of the disparate
impact and disparate treatment of protected-status borrowers. Armed with
a better understanding of the disparities that exist in our financial
system, we can highlight inequities and create a roadmap to improve
equity in access to finance.

Notes
=====

This is the preliminary version and implementation of the ZRP tool.
We\'re dedicated to continue improving both the algorithm and
documentation and hope that government agencies, lenders, citizen data
scientists and other interested parties will help us improve the model.
Details of the model development process can be found in the [model
development documentation](./model_report.rst)

Install
=======

Install requires an internet connection. The package has been tested on python 3.7.4, but should likely work with 3.7.X.

Note: Due to the size and number of lookup tables necesary for the zrp
package, total installation requires 3 GB of available space.

### Setting up your virtual environment

We recommend installing zrp
inside a [python virtual
environment](https://docs.python.org/3/library/venv.html#creating-virtual-environments).

Run the following to build your virtual envrionment:

    python3 -m venv /path/to/new/virtual/environment

Activate your virtual environment:

    source /path/to/new/virtual/environment/bin/activate
 

Ex.:

    python -m venv /Users/joejones/Documents/ZestAI/zrpvenv
    source /Users/joejones/Documents/ZestAI/zrpvenv/bin/activate
    
    
### Unix-like systems

    pip install zrp

After installing via pip, you need to download the lookup tables and
pipelines using the following command: :

    python -m zrp download

### Windows

    pip install pipwin
    pipwin install gdal
    pipwin install fiona

    pip install zrp

After installing via pip, you need to download the lookup tables and
pipelines using the following command: :

    python -m zrp download

If you're experiencing issues with installation, please consult our [Common Issues](https://github.com/zestai/zrp/blob/main/common_issues.rst#manually-installing-lookup-tables-and-pipeline-files) page.

Data
====

### Training Data

The models available in this package were trained on voter registration
data from the states of Florida , Georgia, and North Carolina. Summary
statistics on these datasets and additional datasets used as validation
can be found
[here](https://github.com/zestai/zrp/blob/main/dataset_statistics.txt) .

***Consult the following to download state voter registration data:***

*   [North Carolina](https://www.ncsbe.gov/results-data/voter-registration-data)
*   [Florida](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/UBIG3F)
*   [Alabama](https://www.alabamainteractive.org/sos/voter/voterWelcome.action)
*   [South Carolina](https://www.scvotes.gov/sale-voter-registration-lists)
*   [Georgia](https://sos.ga.gov/index.php/elections/order_voter_registration_lists_and_files)
*   [Louisiana](https://www.sos.la.gov/ElectionsAndVoting/BecomeACandidate/PurchaseVoterLists/Pages/default.aspx)

### American Community Survey (ACS) Data:

The US Census Bureau details that, \"the American Community Survey (ACS)
is an ongoing survey that provides data every year \-- giving
communities the current information they need to plan investments and
services. The ACS covers a broad range of topics about social, economic,
demographic, and housing characteristics of the U.S. population. The
5-year estimates from the ACS are \"period\" estimates that represent
data collected over a period of time. The primary advantage of using
multiyear estimates is the increased statistical reliability of the data
for less populated areas and small population subgroups. The 5-year
estimates are available for all geographies down to the block group
level.\" ( Bureau, US Census. "American Community Survey 5-Year Data
(2009-2019)." Census.gov, 8 Dec. 2021,
<https://www.census.gov/data/developers/data-sets/acs-5year.html>. )

ACS data is available in 1 or 5 year spans. The 5yr ACS data is the most
comprehensive & is available at more granular levels than 1yr data. It
is thus used in this work.

## Model Development and Feature Documentation

Details of the model development process can be found in the [model
development documentation](./model_report.rst) . Details of the human
readable feature definitions as well as feature importances can be found
[here](https://github.com/zestai/zrp/tree/main/zrp/modeling#feature-definitions).

## Usage and Examples

To get started using the ZRP, first ensure the download is complete (as
described above) and xgboost == 1.0.2

Check out the guides in the
[examples](https://github.com/zestai/zrp/tree/main/examples) folder.
Clone the repo in order to obtain the example notebooks and data; this
is not provided in the pip installable package. If you\'re experiencing
issues, first consult our [common issues
guide](https://github.com/zestai/zrp/blob/main/common_issues.rst).

[Here](https://mybinder.org/v2/gh/zestai/zrp/HEAD), we additionally
provide an interactive virtual environment, via Binder, with ZRP
installed. Once you open this link and are taken to the JupyterLab
environment, open up a terminal and run the following: :

    python -m zrp download

Next, we present the primary ways you\'ll use ZRP.

### ZRP Predictions

**Summary of commands:** :

    >>> from zrp import ZRP
    >>> zest_race_predictor = ZRP()
    >>> zest_race_predictor.fit()
    >>> zrp_output = zest_race_predictor.transform(input_dataframe)

**Breaking down key commands** :

    >>> zest_race_predictor = ZRP()

-   **ZRP(pipe\_path=None, support\_files\_path=\"data/processed\",
    key=\"ZEST\_KEY\", first\_name=\"first\_name\",
    middle\_name=\"middle\_name\", last\_name=\"last\_name\",
    house\_number=\"house\_number\",
    street\_address=\"street\_address\", city=\"city\", state=\"state\",
    zip\_code=\"zip\_code\", race=\'race\', proxy=\"probs\",
    census\_tract=None, street\_address\_2=None, name\_prefix=None,
    name\_suffix=None, na\_values=None, file\_path=None, geocode=True,
    bisg=True, readout=True, n\_jobs=49, year=\"2019\", span=\"5\",
    runname=\"test\")**

    -   What it does:
        -   Prepares data to generate race & ethnicity proxies

    You can find parameter descriptions in the [ZRP
    class](https://github.com/zestai/zrp/blob/main/zrp/zrp.py) and it\'s
    [parent
    class](https://github.com/zestai/zrp/blob/main/zrp/prepare/base.py).

```
    >>> zrp_output = zest_race_predictor.transform(input_dataframe)
```

-   **zest\_race\_predictor.transform(df)**
    -   What it does:
        -   Processes input data and generates ZRP proxy predictions.
        -   Attempts to predict on block group, then census tract, then
            zip code based on which level ACS data is found for. If Geo
            level data is unattainable, the BISG proxy is computed. No
            prediction returned if BISG cannot be computed either.

>   -----------------------------------------------------------------------------
>   Parameters   
>   ------------ ----------------------------------------------------------------
>                df : {DataFrame} Pandas dataframe containing input data
>                (see below for necessary columns)
>
>   -----------------------------------------------------------------------------

Input data, **df**, into the prediction/modeling pipeline **MUST**
contain the following columns: first name, middle name, last name, house
number, street address (street name), city, state, zip code, and zest
key. Consult our [common issues
guide](https://github.com/zestai/zrp/blob/main/common_issues.rst) to
ensure your input data is the correct format.

-   Output: A dataframe with the following columns: AAPI AIAN BLACK
    HISPANIC WHITE source\_block\_group source\_zip\_code source\_bisg :

        >>> zrp_output

        =========== =========== =========== =========== =========== =========== ===================== ====================== ==================  
                    AAPI        AIAN        BLACK       HISPANIC    WHITE       source_block_group    source_census_tract    source_zip_code      
        =========== =========== =========== =========== =========== =========== ===================== ====================== ==================  
        ZEST_KEY                                                                                                                                        
        10          0.021916    0.021960    0.004889    0.012153    0.939082    1.0                   0.0                    0.0                    
        100         0.009462    0.013033    0.003875    0.008469    0.965162    1.0                   0.0                    0.0                    
        103         0.107332    0.000674    0.000584    0.021980    0.869429    1.0                   0.0                    0.0                    
        106         0.177411    0.015208    0.003767    0.041668    0.761946    1.0                   0.0                    0.0                    
        109         0.000541    0.000416    0.000376    0.000932    0.997736    1.0                   0.0                    0.0                    
        ...         ...         ...         ...         ...         ...         ...                   ...                    ...                    
        556         NaN         NaN         NaN         NaN         NaN         0.0                   0.0                    0.0                    
        557         NaN         NaN         NaN         NaN         NaN         0.0                   0.0                    0.0                    
        =========== =========== =========== =========== =========== =========== ===================== ====================== ==================  

One of the parameters to the [parent
class](https://github.com/zestai/zrp/blob/main/zrp/prepare/base.py) that
ZRP() inherits from is `file_path`. This parameter allows you to specify
where the `artifacts/` folder is outputted during the run of the ZRP.
Once the run is complete, the `artifacts/` folder will contain the
outputted race/ethnicity proxies and additional logs documenting the
validity of input data. `file_path` **need not** be specified. If it is
not defined, the `artifacts/` folder will be placed in the same
directory of the script running zrp. Subsequent runs will, however,
overwrite the files in `artifacts/`; providing a unique directory path
for `file_path` will avoid this.

ZRP Build
---------

**Summary of commands** :

    >>> from zrp.modeling import ZRP_Build
    >>> zest_race_predictor_builder = ZRP_Build('/path/to/desired/output/directory')
    >>> zest_race_predictor_builder.fit()
    >>> zrp_build_output = zest_race_predictor_builder.transform(input_training_data)

**Breaking down key commands** :

    >>> zest_race_predictor_builder = ZRP_Build('/path/to/desired/output/directory')

-   **ZRP\_Build(file\_path, zrp\_model\_name = \'zrp\_0\',
    zrp\_model\_source =\'ct\')**
    -   What it does:
        -   Prepares the class that builds the new custom ZRP model.

>   -----------------------------------------------------------------------------
>   Parameters   
>   ------------ ----------------------------------------------------------------
>                file_path : {str} The path where pipeline, model, and
>                supporting data are saved.
>
>                zrp_model_name : {str} Name of zrp_model.
>
>                zrp_model_source : {str} Indicates the source of
>                zrp_modeling data to use.
>   -----------------------------------------------------------------------------
>
> You can find more detailed parameter descriptions in the [ZRP\_Build
> class](https://github.com/zestai/zrp/blob/main/zrp/modeling/pipeline_builder.py).
> ZRP\_Build() also inherits initlizing parameters from its [parent
> class](https://github.com/zestai/zrp/blob/main/zrp/prepare/base.py).

    >>> zrp_build_output = zest_race_predictor_builder.transform(input_training_data)

-   **zest\_race\_predictor\_builder.transform(df)**
    -   What it does:
        -   Builds a new custom ZRP model trained off of user input data
            when supplied with standard ZRP requirements including name,
            address, and race
        -   Produces a custom model-pipeline. The pipeline, model, and
            supporting data are saved automatically to
            \"\~/data/experiments/model\_source/data/\" in the support
            files path defined.
        -   The class assumes data is not broken into train and test
            sets, performs this split itself, and outputs predictions on
            the test set.

>   -----------------------------------------------------------------------------
>   Parameters   
>   ------------ ----------------------------------------------------------------
>                df : {DataFrame} Pandas dataframe containing input data
>                (see below for necessary columns)
>
>   -----------------------------------------------------------------------------

Input data, **df**, into this pipeline **MUST** contain the following
columns: first name, middle name, last name, house number, street
address (street name), city, state, zip code, zest key, and race.
Consult our [common issues
guide](https://github.com/zestai/zrp/blob/main/common_issues.rst) to
ensure your input data is the correct format.

-   Output: A dictionary of race & ethnicity probablities and labels.

As mentioned in the ZRP Predict section above, once the run is complete,
the `artifacts/` folder will contain the outputted race/ethnicity
proxies and additional logs documenting the validity of input data.
Similarly, defining `file_path` **need not** be specified, but providing
a unique directory path for `file_path` will avoid overwriting the
[artifacts/]{.title-ref} folder. When running ZRP Build, however,
`artifacts/` also contains the processed test and train data, trained
model, and pipeline.

### Additional Runs of Your Custom Model
After having run ZRP\_Build() you can re-use your custom model just like
you run the packaged model. All you must do is specify the path to the
generated model and pipelines (this path is the same path as
\'/path/to/desired/output/directory\' that you defined previously when
running ZRP\_Build() in the example above; we call this \'pipe\_path\').
Thus, you would run: :

    >>> from zrp import ZRP
    >>> zest_race_predictor = ZRP('pipe_path')
    >>> zest_race_predictor.fit()
    >>> zrp_output = zest_race_predictor.transform(input_dataframe)

Validation
==========

The models included in this package were trained on publicly-available
voter registration data and validated multiple times: on hold out sets
of voter registration data and on a national sample of PPP loan
forgiveness data. The results were consistent across tests: 20-30% more
African Americans correctily identified as African American, and 60%
fewer whites identified as people of color as compared with the status
quo BISG method.

To see our validation analysis with Alabama voter registration data,
please check out [this
notebook](https://github.com/zestai/zrp/blob/main/examples/analysis/Alabama_Case_Study.md).

Performance on the national PPP loan forgiveness dataset was as follows
(comparing ZRP softmax with the BISG method):

*African American*

| Statistic           | BISG  | ZRP   | Pct. Diff |
|---------------------|-------|-------|-----------|
| True Positive Rate  | 0.571 | 0.700 | +23% (F)  |
| True Negative Rate  | 0.954 | 0.961 | +01% (F)  |
| False Positive Rate | 0.046 | 0.039 | -15% (F)  |
| False Negative Rate | 0.429 | 0.300 | -30% (F)  |

*Asian American and Pacific Islander*

| Statistic           | BISG  | ZRP   | Pct. Diff |
|---------------------|-------|-------|-----------|
| True Positive Rate  | 0.683 | 0.777 | +14% (F)  |
| True Negative Rate  | 0.982 | 0.977 | -01% (U)  |
| False Positive Rate | 0.018 | 0.023 | -28% (F)  |
| False Negative Rate | 0.317 | 0.223 | -30% (F)  |

*Non-White Hispanic*

| Statistic           | BISG  | ZRP   | Pct. Diff |
|---------------------|-------|-------|-----------|
| True Positive Rate  | 0.599 | 0.711 | +19% (F)  |
| True Negative Rate  | 0.979 | 0.973 | -01% (U)  |
| False Positive Rate | 0.021 | 0.027 | -29% (F)  |
| False Negative Rate | 0.401 | 0.289 | -28% (F)  |

*White, Non-Hispanic*

| Statistic           | BISG  | ZRP   | Pct. Diff |
|---------------------|-------|-------|-----------|
| True Positive Rate  | 0.758 | 0.906 | +19% (F)  |
| True Negative Rate  | 0.758 | 0.741 | -02% (U)  |
| False Positive Rate | 0.242 | 0.259 | +07% (U)  |
| False Negative Rate | 0.241 | 0.094 | -61% (F)  |

Authors
=======

> -   [Kasey
>     Matthews](https://www.linkedin.com/in/kasey-matthews-datadriven/)
>     (Zest AI Lead)
> -   [Piotr Zak](https://www.linkedin.com/in/piotr-zak-datadriven/) (Algomine)
> -   [Austin Li](https://www.linkedin.com/in/austinwli/) (Harvard T4SG)
> -   [Christien
>     Williams](https://www.linkedin.com/in/christienwilliams/) (Schmidt
>     Futures)
> -   [Sean Kamkar](https://www.linkedin.com/in/sean-kamkar/) (Zest AI)
> -   [Jay Budzik](https://www.linkedin.com/in/jaybudzik/) (Zest AI)

Contributing
============

Contributions are encouraged! For small bug fixes and minor
improvements, feel free to just open a PR. For larger changes, please
open an issue first so that other contributors can discuss your plan,
avoid duplicated work, and ensure it aligns with the goals of the
project. Be sure to also follow the [Code of
Conduct](https://github.com/zestai/zrp/blob/main/CODE_OF_CONDUCT.md).
Thanks!

Maintainers
-----------

Maintainers should additionally consult our documentation on
[releasing](https://github.com/zestai/zrp/blob/main/releasing.rst).
Follow the steps there to push new releases to Pypi and Github releases.
With respect to Github releases, we provide new releases to ensure
relevant pipelines and look up tables requisite for package download and
use are consistently up to date.

Wishlist
========

Support for the following capabilities is planned:

-   add multiracial classification output support
-   national validation datasets and validation partners
-   pointers to additional training data
-   add support for gender and other protected bases

License
=======

The package is released under the [Apache-2.0
License](https://opensource.org/licenses/Apache-2.0).

Results and Feedback
====================

Generate interesting results with the tool and want to share it or other
interesting feedback? Get in touch via <abetterway@zest.ai>.
