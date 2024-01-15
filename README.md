A simple machine learning project on a classification task with a slightly imbalanced target class. 


#### DataSet 
The dataset used here for predicting which passengers survives is from the following Kaggle URL :

https://www.kaggle.com/competitions/titanic

Variables are as follows:

- PassengerId
- Pclass: Ticket class. 1,2,3 for 1st, 2nd and 3rd class, respectively
- Name
- Sex
- Age
- SibSp: number of siblings/spouses on board
- Parch: number of parents/children on board
- Ticket: ticker number
- Fare: passenger fare
- Cabin: cabin number
- Embarked: the ort of embarkation. S=Southampton, C = Cherbourg, Q = Queenstown

- Survived: 0 for no, 1 for yes



#### Conclusions from EDA:
- Fare is quite correlated with survival.
- Having a longer name is slightly more correlated with survival than fare. This agrees with the analysis of "name_length" vs survival. Those from Pclass 1 tend to have a longer name.
- Passengers in cabins with first_char in the early parts of the alphabet are more likely to survive.
- Passengers with missing cabin info are a lot less likely to survive. Those with missing cabin info are also more likely to be in Pclass 3, the lowest Pclass, corresponding to the lowest fares.
- Those from Pclass 1 are more likely to survive compared to those from Pclass 2 or 3.
- Males are less likely to survive.
- Those who embarked from Queenstown are have a higher chance of survival.
- Very young children of <10 years of age are more likely to survive than not. In contrast, a passenger is more likely to die across all other age groups.
- Age, Parch and SibSp are right-skewed.


#### Description of logical steps/flow of the pipeline

1. The raw data is a .csv file, train.csv.
2. Data extraction from .csv files are performed by the script data_extraction.py, giving a pandas dataframe as an output.
3. The dataframe is fed into the script data_preprocessing.py, where features are processed, including filling in of missing values, and synthetic data are generated in the training set using the SMOTEEN algorithm. It is subsequently split into training and testing sets stratified by the ratios of the target class, and the script outputs the train and test data.
3. The data is fed into algo.py containing machine learning algorithms which print classification reports for the classification task and outputs the classifier for each algorithm in this script. 


#### Choice of models and evaluations

- This is a classification problem with an imbalanced binary target class, and the extra trees classifier, random forest classifier and logistic regression are used here. These three models do not have issues with skewed data.  We can look at the f1 score of these models, given that the target classes are imbalanced.

- The choices and evaluations of the above mentioned models are as follows:

1. logistic regression:
    A "saga" solver with L1 regularization is used to reduce the impact of noisy features and focuses on the most relevant features, potentially improving generalization. It is also a simple model where feature importance is easily interpretable.
    Performs well on the majority class, with an f1 score of 0.82. It performs rather poorly on the minority class however, with a f1 score of 0.69.
2. random forest:
    By building multiple decision trees, and considering a random subset of features at each split, this helps to reduce overfitting and improving generalization by making it less sensitive to noise and outliers in individual features. It gives a simpler model compared to extra trees, which might further reduce overfitting and is robust on imbalanced classes.
    Performs well on the majority class, with an f1 score of 0.83. A decent performance on the minority class however, with a f1 score of 0.75.
3. extra tress:
    Similar to random treesm, but instead uses all features at each split, giving a more complex model than random forest. 
    Performs better than logistic regression but worse than random forest, with f1 score of 0.82 and 0.73 for the majority and minority class, respectively.
    
    
#### Feature Importance
Since the random forest performs best, we look at the feature importances. Top 5 important features are the sex(male/female), fare, name_length, age, and Pclass. This agrees with observations from EDA. 



