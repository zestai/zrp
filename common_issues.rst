Why Is ZRP Breaking or Not Returning Proxies?
______________________________________________

In this document, we seek to provide insight for troubleshooting issues you may encounter while running the ZRP. We plan to address these issues in future ZRP updates. Contributions are encouraged! For small bug fixes and minor improvements, feel free to just open a PR. For larger changes, please open an issue first so that other contributors can discuss your plan, avoid duplicated work, and ensure it aligns with the goals of the project. Be sure to also follow the `Code of Conduct <./CODE_OF_CONDUCT.md>`_. Thanks!

#. A provided 'state' in the input dataset is an empty string. This causes an error when geocoding your data.
#. A provided 'state' in the input dataset is not in `inverse state mapping <./zrp/data/processed/inv_state_mapping.json>`_. The 'state' is required to be one of the 50 states in the US. We also use standard 2 letter abbreviations. 
#. ZRP has been run with only first name (or only middle name, last name, or address, respectively) provided. ZRP does not return a proxy when only one of these is provided; all must be a part of the input data frame.
#. There are spaces in the house number, ex.: "103 a". House numbers should not contain spaces.
#. Supplemental BISG proxies cannot be provided if zipcode is not a part of the input data.
#. You have duplicate rows in your data.
#. The current version of the ZRP accepts the following race "labels/classes": 'BLACK', 'HISPANIC', 'AAPI', 'AIAN', 'WHITE'. The input training data to ZRP_Build shouldn't contain any other labels in the provided 'race' column.
