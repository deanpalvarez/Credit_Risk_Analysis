# Credit_Risk_Analysis
# Credit_Risk_Analysis
# Overview
Here we load in a large dataset of credit applications to Python with 80+ columns of lending criteria in order to build/compare multiple machine learning modules in order to test their accuracy. We determine the "loan_status" column as the most important to determine an individuals credit risk, while factoring in the rest of the columns while building machine learning models to determine someone's ability to be approved.

# Results
In layman's terms we isolate the loan status column (our 'y' value) as our data to compare against the rest of the data we run through the machine learning models (our 'x' value). Our 'x' data is run through several different machine learning models; oversampling, undersampling, etc. in multiple different manners. this is all compared against the 'y' data and given an accuracy score, classification report, etc. all to determine how accurate our machine learning models are. 

Our data must first be 'cleaned' and transformed into numerical data, so it can be passed through machine learning algorithms, and allows text to have numerical value. This means literal words in the spreadsheet like "low risk" and "high risk" can be interpreted as 1 or 0, and numerical data such as debt/loan amounts can remain as is. We also get rid of columns we don't need, which can also at the same time further complicate/disrupt our analysis. Part of this process can be shown here:
```
df['home_ownership'] = df['home_ownership'].str.replace('RENT', '0')
df['home_ownership'] = df['home_ownership'].str.replace('ANY', '0')
df['home_ownership'] = df['home_ownership'].str.replace('MORTGAGE', '1')
df['home_ownership'] = df['home_ownership'].str.replace('OWN', '1')
df['home_ownership'] = df['home_ownership'].astype(int)
```
This is done for a number of columns, but this just goes to show how we can turn literal words into numerical data that can be understood/analyzed by a machine learning module.

# Summary
Between oversampling, undersampling, SMOTE/SMOTEEN methods, etc. we received accuracy scores (0 being least/1 being most) of roughly between 0.5-0.6, meaning our machine learning models have done an okay job at predicting credit risk based on our information, but not quite enough to be considered fully functional on its own.

We did however find in a few classification reports, that the precision for low-risk applications was exactly or close to 1.0 where the high-risk precision was closer to 0.5, as shown below using a 'balanced random forest classifier':
```
                  pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.61      0.58      0.01      0.60      0.36        87
   low_risk       1.00      0.58      0.61      0.74      0.60      0.35     17118

avg / total       0.99      0.58      0.61      0.73      0.60      0.35     17205
```
It's worth noting that we have 68,000+ low risk applications and only about 350 high risk applications in our dataset, so we can truthfully conclude that if we we're to run machine learning models on different datasets with more high risk applicants, it would justify the accuracy of the models we've built.