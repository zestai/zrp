ZRP Model Development Documentation
####################################


Problem Statement
==================

To comply with federal fair lending laws, banks and credit unions must prove they don’t discriminate based on race and other protected statuses. But lenders aren’t allowed (except in mortgage lending) to ask the race of the applicant. And, even in mortgage lending, almost a third of applicants put nothing down.

In the absence of data, lenders, regulators and credit bureaus have to guess. The de facto way to do that is with a simple formula called Bayesian Improved Surname Geocoding. The RAND Corporation developed BISG 20 years ago to study discrimination in health care. It brought much-needed objectivity to fair lending analysis and enforcement with a simple formula that combines last name and ZIP code, or Census tract, to calculate the best estimate. RAND said BISG was right at least 9 out of 10 times in identifying people as Black, especially in racially homogenous areas.

The problem is that our country is not racially homogenous, and the predictiveness of surnames gets less accurate every year as neighborhoods diversify and densify, and as the rate of racial intermarriage increases. A 2014 Charles River Associates study on auto loans found BISG correctly identified Black American borrowers 24 percent of the time at an 80 percent confidence threshold. The Consumer Financial Protection Bureau, using a different set of loans, found that BISG correctly identified only 39 percent of Black Americans.

We’re not saying to throw BISG out, but let’s use it only until a better alternative is ready. Data science has advanced since Bayesian algorithms debuted in the 1800s. We should harness the latest tech for good, and there’s some promising work already being done out there. 

Zest’s data science team developed the Zest Race Predictor (ZRP) as a BISG replacement. At its core is a machine-learning model that estimates race using first, middle, and last names and a richer location data set gathered by the US Census.  By using more data:  full name and many more location attributes -- and better math:  gradient boosting -- ZRP significantly improvess the accuracy of race estimation.

Modeling Data
==================


**Names, Addresses, and Class Labels** 

The initial model development dataset includes voter registration data from the states of Florida, North Carolina, and Georgia. Summary statistics on these datasets and additional datasets used as validation can be found `here <https://github.com/zestai/zrp/blob/main/dataset_statistics.txt>`_ . 

Consult the following to download state voter registration data:
 * `North Carolina <https://www.ncsbe.gov/results-data/voter-registration-data>`_
 * `Florida <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/UBIG3F>`_
 * `Alabama <https://www.alabamainteractive.org/sos/voter/voterWelcome.action>`_
 * `South Carolina <https://www.scvotes.gov/sale-voter-registration-lists>`_
 * `Georgia <https://sos.ga.gov/index.php/elections/order_voter_registration_lists_and_files>`_
 * `Louisiana <https://www.sos.la.gov/ElectionsAndVoting/BecomeACandidate/PurchaseVoterLists/Pages/default.aspx>`_

Voter registration data was selected because it is a large, publicly-available database of names, addresses and ground truth labels (self-reported race and ethnicity).  Ideally a more comprehensive list of names addresses and self-reported race/ethnicity from the US Census Bureau would be used to train the model, but such a dataset is not publicly available.


**American Community Survey (ACS) Attributes** 
 
The US Census Bureau details that, "the American Community Survey (ACS) is an ongoing survey that provides data every year -- giving communities the current information they need to plan investments and services. The ACS covers a broad range of topics about social, economic, demographic, and housing characteristics of the U.S. population. The 5-year estimates from the ACS are "period" estimates that represent data collected over a period of time. The primary advantage of using multiyear estimates is the increased statistical reliability of the data for less populated areas and small population subgroups. The 5-year estimates are available for all geographies down to the block group level." ( Bureau, US Census. “American Community Survey 5-Year Data (2009-2019).” Census.gov, 8 Dec. 2021, https://www.census.gov/data/developers/data-sets/acs-5year.html. )

ACS data is available in 1 or 5 year spans. The 5yr ACS data is the most comprehensive & is available at more granular levels than 1yr data. It is thus used in this work. We elaborate below on how ACS data is used.


Model Development
##################

  * **Data Preparation:** Initial dataset definition, sampling, data cleansing, feature creation, target and data selection
  * **Model Training:** Algorithm selection, hyperparameter selection
  * **Model Evaluation:** Model validation, benchmarking and model performance
  


Data Preparation
==================
The modeling process began with data acquisiton. The acquired voter registration data, Census shapefiles, and ACS demographic data contain a super-set of information of the following nature:

* Data used for processing
* Data used for model training
* Data used for model validation
* Data not appropriate for modeling (later excluded or not used)


Overview
____________________

Initial versions of the ZRP were place-specific.  That is, a given zip code was a predictor in the model.  This resulted in a model that was limited to work in the specific places in which it had been trained.  However, not all states release their voter records, and so the challenge was to make a model that could be trained using voter registration data from some small number of states, yet still predict accurately in other unseen geographic areas.

To address this challenge, the next generation ZRP models use Census block group, tract, or zip code attributes.  During the summer of 2021, Harvard undergraduate Austin Li joined the Zest team to develop this next generation of models.  Austin developed a method of geocoding an address to look up its Census block group, tract, or ZIP code by leveraging the Census `ARCGIS TIGER/Line Shapefiles <https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html>`.  The smallest, most granular, matching area (block group, tract, or zip code) is then used to look up attributes of that location in the American Community Survey database, which provides demographic data at the block group, tract and zip code levels.

Many ACS attributes were normalized to percentages of total or to standard statistics (e.g., % of the block group that self-reported they were African American, or median household income for the tract).  By using the normalized attributes of the location instead of the location itself, the model ZRP model can now transfer learnings from one block group, tract or zip code to another and thus operate nationwide.

In order to facilitate fast translation from address to Census block group, tract, or (in the worst case) zip Code, attributes, lookup tables are compiled.

To build the training and test datasets, the voter registration data is joined with ACS attributes via the address matching process described above.  

The full list of predictive variables in the model can be found `here. <https://github.com/zestai/zrp/blob/main/zrp/modeling/feature_definitions.md>`



Data Sampling
____________________
In order to develop the model, representative data with self-reported name, address, and race needed to be acquired. The current ZRP data preparation includes 2021 Florida, Georgia, and North Carolina voter registration datasets. Exploratory data analysis (EDA) exposed data that was not appropriate for modeling. Following EDA each dataset was reduced based on the following filtration criteria records were removed that: requested public record exemption, did not contain adequate address information, exhibited high missingness, non-unique, or did not self-report race or ethnicity. 

The model development dataset is established when treating the voter registration data as one dataset. The model development dataset was split into 4 distinct subsets: one for training, one for internal validation, one for final testing, and a hold out to support ongoing model development. The hold out contains about 30% of the data by state. Aiming for an unbiased representation of the data, we employed random sampling when choosing the dataset splits. The multi-split strategy ensures that the model is not overfitting to the training dataset; that it will be robust to future, unseen data; that the performance is not overstated; and that updates can be implemented. Please refer to the split table below to see the current splits.


+------------------+--------------+-----------------+
| Dataset          | Total Obs    | Total Train Obs | 
+------------------+--------------+-----------------+
| Florida          | 14,215,868   | 5,049,617       | 
+------------------+--------------+-----------------+
| Georgia          | 6,676,561    | 1,942,893       | 
+------------------+--------------+-----------------+
| North Carolina   | 6,586,528    | 2,574,455       | 
+------------------+--------------+-----------------+



Data Summary
____________________
The disaggregated race and ethnicity class information is tabulated below for the training dataset and the United States popultion estimates. 

+---------------------+-------------+---------------+-----------------------+
| Class               | Train Count | Train Percent | National Estimate (%) |
+---------------------+-------------+---------------+-----------------------+
| Asian American and  |             |               |                       |
| Pacific Islander    | 215,866     | 2.3%          | 6.1%                  |
+---------------------+-------------+---------------+-----------------------+
| American Indian     |             |               |                       |
| and Alaskan Native  | 41,872      | 0.4%          | 1.3%                  |
+---------------------+-------------+---------------+-----------------------+
| African American    |             |               |                       |
| or Black            | 2,001,315   | 20.9%         | 13.4%                 |
+---------------------+-------------+---------------+-----------------------+
| Hispanic or Latino  | 1,182,740   | 12.4%         | 18.5%                 |
+---------------------+-------------+---------------+-----------------------+
| White               | 6,125,172   | 64.0%         | 60.1%                 |
+---------------------+-------------+---------------+-----------------------+


Note there was no consistent classification of race identities of multiracial or other so they were not included in model development.

Sample Weights
____________________
Sample weights were consutructed such that proportion of the sample weight associated with each race/ethnicity in the training dataset mimics the national distribution of race/ethnicity. The look-a-like sample weighting was done at the state level.


+-----------------+-----------------+---------------+
| state           | race            | sample_weight |
+-----------------+-----------------+---------------+
| Florida         | WHITE            | 0.9406       |
+-----------------+-----------------+---------------+
| Florida         | BLACK           | 0.9770        |
+-----------------+-----------------+---------------+
| Florida         | AIAN            | 3.9046        |
+-----------------+-----------------+---------------+
| Florida         | HISPANIC        | 0.9565        |
+-----------------+-----------------+---------------+
| Florida         | AAPI            | 2.8882        |
+-----------------+-----------------+---------------+
| Georgia         | WHITE            | 1.1152       |
+-----------------+-----------------+---------------+
| Georgia         | BLACK           | 0.3718        |
+-----------------+-----------------+---------------+
| Georgia         | AAPI            | 1.6984        |
+-----------------+-----------------+---------------+
| Georgia         | HISPANIC        | 3.4281        |
+-----------------+-----------------+---------------+
| Georgia         | AIAN            | 2.6944        |
+-----------------+-----------------+---------------+
| North Carolina  | WHITE            | 0.8509       |
+-----------------+-----------------+---------------+
| North Carolina  | BLACK           | 0.5763        |
+-----------------+-----------------+---------------+
| North Carolina  | AIAN            | 2.1578        |
+-----------------+-----------------+---------------+
| North Carolina  | HISPANIC        | 5.4349        |
+-----------------+-----------------+---------------+
| North Carolina  | AAPI            | 4.0384        |
+-----------------+-----------------+---------------+



Algorithms & Model Training Process
=====================================

The problem of predicting race falls within in the class of problems for which supervised machine learning classification algorithms are used. Supervised machine learning algorithms try to create a functional dependence between data points and a given target variable. In this case, the algorithms created a functional dependence between data related to an individual’s name as well as his/her address, and their race/ethnicity.  Classification algorithms try to predict a finite number of target choices; for instance: Black, White, Hispanic, AAPI, AIAN, or Multiracial.

Classification models can be classified according to the mathematical form of the underlying prediction function: linear and non-linear models. In linear models, the separation between distinct classes, or the relationship between different continuous variables, can be modeled using a linear function. Logistic regression, traditionally used for credit modeling, is an example of a linear model, while decision trees and neural networks are non-linear models.

Several types of classification models could be used to address the problem of predicting race. The pros and cons of several options are 
ed in the table below.

+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+
|              Model Type             |                                                               Benefits                                                              |                                   Limitations                                   |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+
|                                     |                                                                                                                                     | High bias                                                                       |
|                                     | Low variance                                                                                                                        | Underperforms when feature space is large                                       |
| Logistic Regression                 | Easy to interpret                                                                                                                   | Relies on transformation for non-linear features                                |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+
|                                     | Computationally fast                                                                                                                | Relies on independence assumption; will perform badly if assumption breaks down |
|                                     | Simple to implement                                                                                                                 |                                                                                 |
| Naive Bayes                         | Works well with high dimensions                                                                                                     |                                                                                 |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+
|                                     | Performs similarly to logistic regression with linear boundary                                                                      | Susceptible to overfitting depending on kernel                                  |
|                                     | Performs well with non-linear boundary depending on the kernel                                                                      | Sensitive to outliers                                                           |
| Support Vector Machine (SVM)        | Handles high dimensional data well                                                                                                  | Not very efficient with large number of observations                            |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+--+--+
|                                     | Reduced variance in comparison with simpler tree models                                                                             | Not as easy as simpler trees to visually interpret                              |  |  |
|                                     | Decorrelates trees                                                                                                                  | Trees do not learn from each other                                              |  |  |
| Random Forest                       | Handles categorial and real-valued features well                                                                                    |                                                                                 |  |  |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+
|                                     | Handles missing values easily without preprocessing                                                                                 | Susceptible to overfitting if number of trees is too large                      |  |  |
| Extreme Gradient Boosting (XGBoost) | Highly performant and executes quickly                                                                                              |                                                                                 |  |  |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+
|                                     |                                                                                                                                     | Many parameters to tune                                                         |
| Neural Network                      | Excellent performance on highly complex problems, such as image classification, natural language processing, and speech recognition | Sensitive to missing data and non-standardized features                         |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+

Bayseian and linear models were ruled out as the variables (income, education attainment) are not independent, and the decision surface is not linear.  Random forest was also ruled out due to the better performance from XGBoost that is by now well-known.

XGBoost is a tree model based on a boosting algorithm. It reduces variance and also reduces bias. XGBoost reduces variance because it uses multiple models, by bagging like a Random Forest, but simultaneously reduces bias by training the subsequent model based on the errors by previous models. Since XGBoost sequentially learns from the previous models, it often outperforms Random Forest. The model also has the benefits of Random Forest, which is randomizing the sample to reduce variance.

The biggest concern associated with XGBoost models is overfitting. Therefore, it is important to tune the hyperparameters to make sure the model is not overfitted to the Training Dataset and that it exhibits similar performance on both the Training and OOT Datasets. 

While tree-based models excel on tabular data like we have here, Neural Networks can handle even more complex problems, yet neural networks come with addiitional complexity.   Due to the tabular nature of the data, and keepiing things simple, we selected XGBoost for the ZRP.  A neural network algorithm would be more appropriate if we were considering pictures of people in addition to tabular attributes.


Feature engineering
____________________
The feature engineering pipeline takes name and ACS features as input to prepare data for model build or race predictions (also refered to as race proxies). First, the data is reduced to required modeling features using 'Drop Features'. Next compound last names are handled by splitting compound last names across n rows. Let's take a look at an example if person is named Farrah A. Len-Doe, the input to 'Compound Name FE' will be one dedicated record, as seen below:   


+----------+------------+-------------+-----------+--------------+----------------+----------+--------+----------+
| ZEST_KEY | first_name | middle_name | last_name | house_number | street_address | city     | state  | zip_code |
+----------+------------+-------------+-----------+--------------+----------------+----------+--------+----------+
| Z00100   | Farrah     | A.          | Len-Doe   | 123          | N main st      | burbank  | ca     | 91505    |
+----------+------------+-------------+-----------+--------------+----------------+----------+--------+----------+



That expands to two rows with unique last name values per row.


+----------+------------+-------------+-----------+--------------+----------------+----------+--------+----------+
| ZEST_KEY | first_name | middle_name | last_name | house_number | street_address | city     | state  | zip_code |
+----------+------------+-------------+-----------+--------------+----------------+----------+--------+----------+
| Z00100   | Farrah     | A.          | Len       | 123          | N main st      | burbank  | ca     | 91505    |
+----------+------------+-------------+-----------+--------------+----------------+----------+--------+----------+
| Z00100   | Farrah     | A.          | Doe       | 123          | N main st      | burbank  | ca     | 91505    |
+----------+------------+-------------+-----------+--------------+----------------+----------+--------+----------+


After compound last names are handled, 'App FE' executes general name feature engineering. 'MultiLabelBinarizer` is used to convert the set of targets to, an array-like object, a binary matrix indicating the presence of a class. Targets associated with each record are one hot encoded using 'MultiLabelBinarizer`. Then first, middle and last name are encoded using 'TargetEncoder'. "For the case of categorical target: features are replaced with a blend of posterior probability of the target given particular categorical value and the prior probability of the target over all the training data."( `ref <https://contrib.scikit-learn.org/category_encoders/targetencoder.html>`_). Next the pipeline focuses on engineering of the ACS features. 'CustomRatios' generates ratios, percents, and linear combinations of select ACS features. After generating ACS engineered features, the pipelie resolves the many-to-one data created by the 'Compound Name FE' step by aggregating across expected name columns, at the unique key level. At this point all geo-specific features, like block group, tract, and zip code are no-longer in the feature space. Missing values are imputed using mean, for all numeric features. Lastly, the training dataset's least missing, most unique features with the highest variance and importance are selected. 


Model Creation
____________________

XGBoost 1.0.2 was used to train the model with the following hyperparameters:


+---------------------+----------------------+
| Parameter Name.     | Value.               |
+---------------------+----------------------+
| 'gamma'             | 5                    |
+---------------------+----------------------+
| 'learning_rate'     | 0.01                 |
+---------------------+----------------------+
| 'max_depth'         | 3                    |
+---------------------+----------------------+
| 'min_child_weight'  | 500                  |
+---------------------+----------------------+
| 'n_estimators'      | 2000                 |
+---------------------+----------------------+
| 'subsample'         | 0.8                  |
+---------------------+----------------------+
| 'objective'         | multi:softprob       |
+---------------------+----------------------+


Around 9.5 million names, locations, and self-reported race/ethnicities from the 2021 Florida, Georgia and North Carolina voter registration database were set aside for training.

Several models are trained:  one for Census block group, one for Census tract, and another for the zip code. 


Prediction Process
____________________

The inputs to ZRP include name and address.  The address is used to lookup attributes of the correpsonding region.  The lookup process starts with retrieval of Census block group attributes.  If the block group lookup fails, then Census tract attributes are retrieved.  If the Census tract lookup fails, then ZIP code attributes are retrieved.  ACS attributes associated with the retrieved geographic area are then appended to the first, middle, and last name.  The resulting vector of predictors is then used as input to the corresponding model (e.g., block group, tract, or zip code-based model).


Model Evaluation
==================


A validation dataset was constructed using 2021 Alabama voter registration data comprised of about 235,000 randomly sampled records. Around 230,000 records had appropriate data for generating race predictions. Please refer to the *Data Sampling* section to review filtration criteria. The race and ethnicity class information is tabulated below for the Alabama validation dataset. The table include United States popultion estimates by race and ethnicity, these estiamtes are not indicative of the true registered voter population.


+---------------------+----------------+-----------------------+
| Class               | Sample Percent | National Estimate (%) |
+---------------------+----------------+-----------------------+
| Asian American and  |                |                       |
| Pacific Islander    |  1.1%          | 1.6%                  |
+---------------------+----------------+-----------------------+
| American Indian     |                |                       |
| and Alaskan Native  |  0.3%          | 0.7%                  |
+---------------------+----------------+-----------------------+
| African American    |                |                       |
| or Black            |  23.6%         | 26.8%                 |
+---------------------+----------------+-----------------------+
| Hispanic or Latino  |  2.5%          | 4.6%                  |
+---------------------+----------------+-----------------------+
| White               |  72.6%         | 65.3%                 |
+---------------------+----------------+-----------------------+


The benchmark model used for comparison in this section is BISG. Across the board, with significant class sizes, we can see ZRP outperform BISG. BISG falls short when proxying race or ethnicity of minority groups exhibited by low TPRs across  minority classes. Predictive performance of the ZRP model on the Alabama validation dataset is shown below:

**BLACK** (African American)

+----------+-----------+-----------+-----------+
| Stat.    | ZRP       | BISG      | Pct Diff  |
+----------+-----------+-----------+-----------+
| TPR      | 0.738314  | 0.569785  | 25.77%    |
+----------+-----------+-----------+-----------+
| TNR      | 0.963988  | 0.907395  | 6.05%     |
+----------+-----------+-----------+-----------+
| FPR      | 0.036012  | 0.092605  | -88.0%    |
+----------+-----------+-----------+-----------+
| FNR      | 0.261686  | 0.430215  | -48.71%   |
+----------+-----------+-----------+-----------+
| PPV      | 0.863487  | 0.654969  | 27.46%    |
+----------+-----------+-----------+-----------+
| AUC      | 0.851151  | 0.73859   | 14.16%    |
+----------+-----------+-----------+-----------+


**AAPI** (Asian American and Pacific Islander)

+----------+-----------+-----------+-----------+
| Stat.    | ZRP       | BISG      | Pct Diff  |
+----------+-----------+-----------+-----------+
| TPR      | 0.665479  | 0.531275  | 22.43%    |
+----------+-----------+-----------+-----------+
| TNR      | 0.996707  | 0.998798  | -0.21%    |
+----------+-----------+-----------+-----------+
| FPR      | 0.003293  | 0.001202  | 93.05%    |
+----------+-----------+-----------+-----------+
| FNR      | 0.334521  | 0.468725  | -33.42%   |
+----------+-----------+-----------+-----------+
| PPV      | 0.692054  | 0.83096   | -18.24%   |
+----------+-----------+-----------+-----------+
| AUC      | 0.831093  | 0.765036  | 8.28%     |
+----------+-----------+-----------+-----------+

**WHITE** (White, non-Hispanic)

+----------+-----------+-----------+-----------+
| Stat.    | ZRP       | BISG      | Pct Diff  |
+----------+-----------+-----------+-----------+
| TPR      | 0.947022  | 0.846848  | 11.17%    |
+----------+-----------+-----------+-----------+
| TNR      | 0.761921  | 0.634041  | 18.32%    |
+----------+-----------+-----------+-----------+
| FPR      | 0.238079  | 0.365959  | -42.34%   |
+----------+-----------+-----------+-----------+
| FNR      | 0.052978  | 0.153152  | -97.2%    |
+----------+-----------+-----------+-----------+
| PPV      | 0.91339   | 0.859847  | 6.04%     |
+----------+-----------+-----------+-----------+
| AUC      | 0.854471  | 0.740444  | 14.3%     |
+----------+-----------+-----------+-----------+


**HISPANIC**  

+----------+-----------+-----------+-----------+
| Stat.    | ZRP       | BISG      | Pct Diff  |
+----------+-----------+-----------+-----------+
| TPR      | 0.852894  | 0.502213  | 51.76%    |
+----------+-----------+-----------+-----------+
| TNR      | 0.987567  | 0.990625  | -0.31%    |
+----------+-----------+-----------+-----------+
| FPR      | 0.012433  | 0.009375  | 28.05%    |
+----------+-----------+-----------+-----------+
| FNR      | 0.147106  | 0.497787  | -108.76%  |
+----------+-----------+-----------+-----------+
| PPV      | 0.633697  | 0.57464   | 9.77%     |
+----------+-----------+-----------+-----------+
| AUC      | 0.920231  | 0.746419  | 20.86%    |
+----------+-----------+-----------+-----------+

**AIAN** (Native American)

+----------+-----------+-----------+-----------+
| Stat.    | ZRP       | BISG      | Pct Diff  |
+----------+-----------+-----------+-----------+
| TPR      | 0.041739  | 0.040000  | 4.26%     |
+----------+-----------+-----------+-----------+
| TNR      | 0.998926  | 0.999716  | -0.08%    |
+----------+-----------+-----------+-----------+
| FPR      | 0.001074  | 0.000284  | 116.4%    |
+----------+-----------+-----------+-----------+
| FNR      | 0.958261  | 0.960000  | -0.18%    |
+----------+-----------+-----------+-----------+
| PPV      | 0.088889  | 0.261364  | -98.49%   |
+----------+-----------+-----------+-----------+
| AUC      | 0.520333  | 0.519858  | 0.09%     |
+----------+-----------+-----------+-----------+


Model Limitations
==================

This model is designed to predict race/ethnicity based on names and addresses of people residing in the United States only.





