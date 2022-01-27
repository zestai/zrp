Problem Statement
__________________

To comply with federal fair lending laws, banks and credit unions must prove they don’t discriminate based on race and other protected statuses. But lenders aren’t allowed (except in mortgage lending) to ask the race of the applicant. And, even in mortgage lending, almost a third of applicants put nothing down.

In the absence of data, lenders, regulators and credit bureaus have to guess. The de facto way to do that is with a simple formula called Bayesian Improved Surname Geocoding. The RAND Corporation developed BISG 20 years ago to study discrimination in health care. It brought much-needed objectivity to fair lending analysis and enforcement with a simple formula that combines last name and ZIP code, or Census tract, to calculate the best estimate. RAND said BISG was right at least 9 out of 10 times in identifying people as Black, especially in racially homogenous areas.

The problem is that our country is not racially homogenous, and the predictiveness of surnames gets less accurate every year as neighborhoods diversify and densify, and as the rate of racial intermarriage increases. A 2014 Charles River Associates study on auto loans found BISG correctly identified Black American borrowers 24 percent of the time at an 80 percent confidence threshold. The Consumer Financial Protection Bureau, using a different set of loans, found that BISG correctly identified only 39 percent of Black Americans.

We’re not saying to throw BISG out, but let’s use it only until a better alternative is ready. Data science has advanced since Bayesian algorithms debuted in the 1800s. We should harness the latest tech for good, and there’s some promising work already being done out there. 

Zest’s data science team, the Zest Race Predictor (ZRP) as a BISG replacement. At its core is a machine-learning model that estimates race using first, middle, and last names and a richer location data set.

Modeling Data
______________

Training Data
==============
The initial model development dataset includes voter registration data from the states of Florida and North Carolina. Summary statistics on these datasets and additional datasets used as validation can be found `here <./dataset_statistics.txt>`_ . 

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

ACS data is available in 1 or 5 year spans. The 5yr ACS data is the most comprehensive & is available at more granular levels than 1yr data. It is thus used in this work. We elaborate below on how ACS data is used.


Model Development
__________________

  * **Data Preparation:** Initial dataset definition, data validation and analysis, segmentation and sampling, data cleansing, feature creation, target and data selection
  * **Model Training:** Algorithm selection, hyperparameter tuning and optimization
  * **Model Evaluation:** Model validation, benchmarking approach, model performance, difference drivers, model stability and model limitations

Data Preparation
_________________

Follow up with Kasey to see if any of the stuff in the MRM were carried out here (summary stats, predictive power, data sampling, variables used, target analysis)

Algorithms & Model Training
________________

The problem of predicting race falls within in the class of problems for which supervised machine learning algorithms are used. Supervised machine learning algorithms try to create a functional dependence between data points and a given target variable. In this case, the algorithms created a functional dependence between data related to an individual’s name as well as his/her address, and their race.

Supervised machine learning algorithms can be further subdivided into regression and classification algorithms. Regression algorithms are designed to handle continuously varying target variables such as the amount of credit that ought to be issued. Classification algorithms try to predict among a finite discrete number of target choices; for instance: Black, White, Hispanic, AAPI, AIAN, or Multiracial.

Classification models can also be classified according to the mathematical form of the underlying prediction function: linear and non-linear models. In linear models, the separation between distinct classes, or the relationship between different continuous variables, can be modeled using a linear function. Logistic regression, traditionally used for credit modeling, is an example of a linear model, while decision trees and neural networks are non-linear models.

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

Naive Bayes models rely on strong independence assumptions and are usually outperformed by other models, so it was not considered for the ZRP. SVM is a linear separator, so when the data is not linearly separable in its original form, a kernel must be used to project data into high dimensional space for segmentation. In many instances, a generic radial basis function (RBF) kernel is used. Other algorithms, however, have shown to out perform SVM due to techniques that better adapt to data compared to the generic kernel. Logistic regression was also not considered, as it assumes that there is one smooth linear decision boundary; it is very difficult to transform or engineer the features to satisfy that assumption. Tree-based models can be expected to outperform logistic regression when well-tuned. Random Forest is a tree model based on a bagging algorithm. It reduces variance by creating many different independant submodels through resampling both the column and row space, making the resulting collective model more robust. 

XGBoost is a tree model based on a boosting algorithm. It reduces variance and also reduces bias. XGBoost reduces variance because it uses multiple models, by bagging like a Random Forest, but simultaneously reduces bias by training the subsequent model based on the errors by previous models. Since XGBoost sequentially learns from the previous models, it often outperforms Random Forest. The model also has the benefits of Random Forest, which is randomizing the sample to reduce variance.

The biggest concern of Random Forest and XGBoost models is overfitting. Therefore, it is important to tune the hyperparameters to make sure the model is not overfitted to the Training Dataset thereby exhibiting similar performance on both the Training and OOT Datasets. Hyperparameter tuning for the Zest Model is discussed further below.

Neural Networks can handle very complex problems but are very difficult to initiate. There can be millions of parameters – each of which requires tuning – in a simple Neural Network. These parameters can include a number of hidden layers, inputs or outputs of each hidden layer, optimization algorithms, activation functions – all of which make it difficult to optimize the Neural Network.

Data Pipeline
=============

For model training, we input dataframes consisting of ____________-. From the input, we map an individuals address to their census tract via computed geo lookup tables. We match on this census tract to derive an additional set of demographic data to add to the feature vector. Thus, __________________ are ultimately the predictive variables used in the model.

We one hot encode the race labels. Using each bit of the encoding as a unique target, we then treat first, middle, and last names as categorical features, and target encode these features. In target encoding, features are replaced by a function of the posterior probability of the target (a race label) given a particular categorical value (first, middle, or last name), and the prior probability of the target over all the training data. Ultimately, an individual's name is represented in the final feature vector as multiple target encodings. 

For inference, like with training, we feed in dataframes as input, addresses are mapped to census tracts and thus ACS demographic information, and feature vectors are built from these data and encoded name features. ........

Model Outputs
=============


Model Performance
__________________

