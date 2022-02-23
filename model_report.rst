ZRP Model Development Documentation
====================================


Problem Statement
__________________

To comply with federal fair lending laws, banks and credit unions must prove they don’t discriminate based on race and other protected statuses. But lenders aren’t allowed (except in mortgage lending) to ask the race of the applicant. And, even in mortgage lending, almost a third of applicants put nothing down.

In the absence of data, lenders, regulators and credit bureaus have to guess. The de facto way to do that is with a simple formula called Bayesian Improved Surname Geocoding. The RAND Corporation developed BISG 20 years ago to study discrimination in health care. It brought much-needed objectivity to fair lending analysis and enforcement with a simple formula that combines last name and ZIP code, or Census tract, to calculate the best estimate. RAND said BISG was right at least 9 out of 10 times in identifying people as Black, especially in racially homogenous areas.

The problem is that our country is not racially homogenous, and the predictiveness of surnames gets less accurate every year as neighborhoods diversify and densify, and as the rate of racial intermarriage increases. A 2014 Charles River Associates study on auto loans found BISG correctly identified Black American borrowers 24 percent of the time at an 80 percent confidence threshold. The Consumer Financial Protection Bureau, using a different set of loans, found that BISG correctly identified only 39 percent of Black Americans.

We’re not saying to throw BISG out, but let’s use it only until a better alternative is ready. Data science has advanced since Bayesian algorithms debuted in the 1800s. We should harness the latest tech for good, and there’s some promising work already being done out there. 

Zest’s data science team developed the Zest Race Predictor (ZRP) as a BISG replacement. At its core is a machine-learning model that estimates race using first, middle, and last names and a richer location data set gathered by the US Census.  By using more data:  full name and many more location attributes -- and better math:  gradient boosting -- ZRP significantly improvess the accuracy of race estimation.

Modeling Data
______________


**Names, Addresses, and Class Labels** 

The initial model development dataset includes voter registration data from the states of Florida and North Carolina. Summary statistics on these datasets and additional datasets used as validation can be found `here <./dataset_statistics.txt>`_ . 

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
__________________

  * **Data Preparation:** Initial dataset definition, segmentation and sampling, data cleansing, feature creation, target and data selection
  * **Model Training:** Algorithm selection, hyperparameter selection
  * **Model Evaluation:** Model validation, benchmarking and model performance

Data Preparation
_________________

Initial versions of the ZRP were place-specific.  That is, a given ZIP code was a predictor in the model.  This resulted in a model that was limited to work in the specific places in which it had been trained.  However, not all states release their voter records, and so the challenge was to make a model that could be trained using voter registration data from some small number of states, yet still predict accurately in other unseen geographic areas.

To address this challenge, the next generation ZRP models use Census block group, tract, or ZIP code attributes.  During the summer of 2021, Harvard undergraduate Austin Li joined the Zest team to develop this next generation of models.  Austin developed a method of geocoding an address to look up its Census block group, tract, or ZIP code by leveraging the Census `ARCGIS TIGER/Line Shapefiles <https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html>`.  The smallest, most granular, matching area (block group, tract, or ZIP code) is then used to look up attributes of that location in the American Community Survey database, which provides demographic data at the block group, tract and ZIP code levels.

ACS attributes were normalized to percentages of total or to standard statistics (e.g., % of the block group that self-reported they were African American, or median household income for the tract).  By using the normalized attributes of the location instead of the location itself, the model ZRP model can now transfer learnings from one block group, tract or ZIP code to another and thus operate nationwide.

In order to facilitate fast translation from address to Census block group, tract, or (in the worst case) ZIP Code, attributes, lookup tables are compiled.

To build the training and test datasets, the voter registration data is joined with ACS attributes via the address matching process described above.  

The full list of predictive variables in the model can be found `here. <https://github.com/zestai/zrp/blob/main/zrp/modeling/feature_definitions.md>`



Algorithms & Model Training
________________

The problem of predicting race falls within in the class of problems for which supervised machine learning classification algorithms are used. Supervised machine learning algorithms try to create a functional dependence between data points and a given target variable. In this case, the algorithms created a functional dependence between data related to an individual’s name as well as his/her address, and their race/ethnicity.  Classification algorithms try to predict a finite number of target choices; for instance: Black, White, Hispanic, AAPI, AIAN, or Multiracial.

Classification models can be classified according to the mathematical form of the underlying prediction function: linear and non-linear models. In linear models, the separation between distinct classes, or the relationship between different continuous variables, can be modeled using a linear function. Logistic regression, traditionally used for credit modeling, is an example of a linear model, while decision trees and neural networks are non-linear models.

Several types of classification models could be used to address the problem of predicting race. The pros and cons of several options are listed in the table below.

+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+--+--+
|              Model Type             |                                                               Benefits                                                              |                                   Limitations                                   |  |  |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+--+--+
|                                     |                                                                                                                                     | High bias                                                                       |  |  |
|                                     | Low variance                                                                                                                        | Underperforms when feature space is large                                       |  |  |
| Logistic Regression                 | Easy to interpret                                                                                                                   | Relies on transformation for non-linear features                                |  |  |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+--+--+
|                                     | Computationally fast                                                                                                                | Relies on independence assumption; will perform badly if assumption breaks down |  |  |
|                                     | Simple to implement                                                                                                                 |                                                                                 |  |  |
| Naive Bayes                         | Works well with high dimensions                                                                                                     |                                                                                 |  |  |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+--+--+
|                                     | Performs similarly to logistic regression with linear boundary                                                                      | Susceptible to overfitting depending on kernel                                  |  |  |
|                                     | Performs well with non-linear boundary depending on the kernel                                                                      | Sensitive to outliers                                                           |  |  |
| Support Vector Machine (SVM)        | Handles high dimensional data well                                                                                                  | Not very efficient with large number of observations                            |  |  |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+--+--+
|                                     | Reduced variance in comparison with simpler tree models                                                                             | Not as easy as simpler trees to visually interpret                              |  |  |
|                                     | Decorrelates trees                                                                                                                  | Trees do not learn from each other                                              |  |  |
| Random Forest                       | Handles categorial and real-valued features well                                                                                    |                                                                                 |  |  |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+--+--+
|                                     | Handles missing values easily without preprocessing                                                                                 | Susceptible to overfitting if number of trees is too large                      |  |  |
| Extreme Gradient Boosting (XGBoost) | Highly performant and executes quickly                                                                                              |                                                                                 |  |  |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+--+--+
|                                     |                                                                                                                                     | Many parameters to tune                                                         |  |  |
| Neural Network                      | Excellent performance on highly complex problems, such as image classification, natural language processing, and speech recognition | Sensitive to missing data and non-standardized features                         |  |  |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------+--+--+

Bayseian and linear models were ruled out as the variables (income, education attainment) are not independent, and the decision surface is not linear.  Random forest was also ruled out due to the better performance from XGBoost that is by now well-known.

XGBoost is a tree model based on a boosting algorithm. It reduces variance and also reduces bias. XGBoost reduces variance because it uses multiple models, by bagging like a Random Forest, but simultaneously reduces bias by training the subsequent model based on the errors by previous models. Since XGBoost sequentially learns from the previous models, it often outperforms Random Forest. The model also has the benefits of Random Forest, which is randomizing the sample to reduce variance.

The biggest concern associated with XGBoost models is overfitting. Therefore, it is important to tune the hyperparameters to make sure the model is not overfitted to the Training Dataset and that it exhibits similar performance on both the Training and OOT Datasets. 

While tree-based models excel on tabular data like we have here, Neural Networks can handle even more complex problems, yet neural networks come with addiitional complexity.   Due to the tabular nature of the data, and keepiing things simple, we selected XGBoost for the ZRP.  A neural network algorithm would be more appropriate if we were considering pictures of people in addition to tabular attributes.

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


4,517,348 names, locations, and self-reported race/ethnicities from the Florida and North Carolina voter registration database were used for training.

Sample weights were consutructed such that proportion of the sample weight associated with each race/ethnicity in the training dataset matches the national distribution of race/ethnicity.


Model Evaluation
________________


A hold out dataset was constructed using Alabama voter registration data.  Predictive performance on the Alabama hold out dataset is shown below:

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
_________________

This model is designed to predict race/ethnicity based on names and addresses of people residing in the United States only.





