Zest Race Predictor
____________________

This tool allows users to predict race by only providing an individual's name and address as inputs. The nuance that this tool exploits, however, that gives it far superior accuracy to the Bayesian Improved Surname Geocoding (BISG) tool used by fair lending institutions today, is use of American Community Survey (ACS) data. By cross referencing with our ACS data lookup tables, we've trained models with data as low fedelity as an individuals census block group. Additionally, using ACS data, we've bolstered our training input feature vectors with additional insights such as percentages of a racial group in a census tract, or average houshold income of a census tract. 

Notes
_____

This is the preliminary version and implementation of the ZRP tool. We're dedicated to continue improving both the algorithm and documentation. 


Install
_______

We recommend installing zrp inside a `python virtual environment <https://docs.python.org/3/library/venv.html#creating-virtual-environments>`_.
::

 pip install zrp

Or:
::

 condaa install -c ___________ zrp

After installing via pip, you need to download the lookup tables using the following command:
::

 python -m zrp download

Note: Due to the size and number of lookup tables necesary for the zrp package, total installation requires ___ GB of available space.


Data
_____

Training Data
==============
The models available in this package were trained on voter registration data from the states of Florida and North Carolina. Summary statistics on these datasets and additional datasets used as validation can be found `here <./dataset_statistics.txt>`_ . 

Consult the following to download state voter registration data:
 * `North Carolina <https://www.ncsbe.gov/results-data/voter-registration-data>`_
 * `Florida <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/UBIG3F>`_
 * `Alabama <https://www.alabamainteractive.org/sos/voter/voterWelcome.action>`_
 * `South Carolina <https://www.scvotes.gov/sale-voter-registration-lists>`_
 * `Georgia <https://sos.ga.gov/index.php/elections/order_voter_registration_lists_and_files>`_
 * `Louisiana <https://www.sos.la.gov/ElectionsAndVoting/BecomeACandidate/PurchaseVoterLists/Pages/default.aspx>`_

American Community Survey (ACS) Data:
=====================================
 
The US Census Bureau details that, "the American Community Survey (ACS) is an ongoing survey that provides data every year -- giving communities the current information they need to plan investments and services. The ACS covers a broad range of topics about social, economic, demographic, and housing characteristics of the U.S. population. The 5-year estimates from the ACS are "period" estimates that represent data collected over a period of time. The primary advantage of using multiyear estimates is the increased statistical reliability of the data for less populated areas and small population subgroups. The 5-year estimates are available for all geographies down to the block group level." ( Bureau, US Census. “American Community Survey 5-Year Data (2009-2019).” Census.gov, 8 Dec. 2021, https://www.census.gov/data/developers/data-sets/acs-5year.html. )

ACS data is available in 1 or 5 year spans. The 5yr ACS data is the most comprehensive & is available at more granular levels than 1yr data. It is thus used in this work.


The Models and API
__________

The ZRP can be broken down into four main segments: preprocessing, geocoding, American Community Survey(ACS) integration, and modeling/predictions.



Module Usage
___________

* models
* functions
* data access
^and file structure/locations for these


CLI Usage
__________


Examples
_________

To illustrate how the package can be used, we ...


Authors
_______

 * Kasey Matthews (Zest AI Lead)
 * Austin Li (Harvard T4SG)
 * Christien Williams (Schmidt Futures)
 * Sean Kamkar (Zest AI)
 * Jay Budzik (Zest AI)

Contributing
_____________

Contributions are encouraged! For small bug fixes and minor improvements, feel free to just open a PR. For larger changes, please open an issue first so that other contributors can discuss your plan, avoid duplicated work, and ensure it aligns with the goals of the project. Be sure to also follow the `Code of Conduct <./CODE_OF_CONDUCT.md>`_. Thanks!

Wishlist
__________

Support for the following capabilities is planned:

- ...nothing right now! (Got an idea? Submit an issue/PR!)

License
_________

The package is released under the `Apache-2.0
License <https://opensource.org/licenses/Apache-2.0>`__.

Results and Feedback
_____________________

Generate interesting results with the tool and want to share it or other interesting feedback? Get in touch via abetterway@zest.ai. 
