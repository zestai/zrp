Feature Definitions
---------------------

In files such as `feature_list_bg <./feature_list_bg.json>`_ and `acs_scaler.py <./src/acs_scaler.py>`_, you'll observe features used by the model that are not human readable. These features come from American Community Survey American Community Survey data (`Census Data API Variables <https://api.census.gov/data/2019/acs/acs5/variables.html>`_). We thus mimic the variable names for the corresponding features. You can refer to `Census Data API Variables <https://api.census.gov/data/2019/acs/acs5/variables.html>`_ or our `Feature Definitions Table <./feature_definitions.md>`_ to discover the human readable feature definitions. Please note that the American Community Survey variables use a naming convention that ends with ‘E', and the ZRP doesn’t append this 'E' in our feature_names. Feature importance disaggregated by race, for the `block group model <./block_group_model_feature_importance.xlsx>`_, `census tract model <./census_tract_model_feature_importance.xlsx>`_, and `zip code model <./zip_code_model_feature_importance.xlsx>`_, respectively, is also provided.  


