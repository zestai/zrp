Feature Definitions
---------------------

In files such as `feature_list_bg <./feature_list_bg.json>`_ and `acs_scaler.py <./src/acs_scaler.py>`_, you'll observe features used by the model that are not human readable. These features come from American Community Survey American Community Survey data (`Census Data API Variables <https://api.census.gov/data/2019/acs/acs5/variables.html>`_). We thus mimic the variable names for the corresponding features. Please refer to `Census Data API Variables <https://api.census.gov/data/2019/acs/acs5/variables.html>`_ to discover the human readable feature definitions. Please note that the American Community Survey variables use a naming convention that ends with ‘E', and the ZRP doesn’t append this 'E' in our feature_names. 


