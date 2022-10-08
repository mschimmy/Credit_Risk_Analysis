# Credit Risk Analysis

## Overview of the Analysis

The client, Fast Lending, is a peer-to-peer lending service company that requests a machine learning algorithm be built to predict credit risk. Credit risk is an unbalanced classification problem because good loans typically outnumber bad or risky loans. Different machine learning models with unbalanced classes can be built to resample the data and address the imbalance before making predictions.

### Purpose

The purpose of this analysis is to build and perform multiple machine learning algorithms that predict credit risk. These machine learning models either oversample or undersample the data, or use a combination approach of over- and undersampling to balance classifications, reduce biases, and predict credit risk. The performance of each model is evaluated and the analysis concludes with a recommendation on which model the client should use to predict credit risk.

The dataset used in the analysis is provided by the client and includes information on individual loan applications. Some metrics collected for each application include the loan amount, interest rate, number of installments, loan issue date, and loan status. The variable of interest for the machine learning algorithms is the loan status, which classifies the loans as either low-risk or high-risk.

## Results

### Data Preprocessing

Before building each machine learning model, the dataset is split into features "X" (input) and the target "y" (output). For this analysis, the target variable is the "loan_status" of each application. The features and target are each split into training and testing sets (X_train, X_test, y_train, and y_test) using sklearn's train_test_split module.

![Splitting the training and testing sets](https://github.com/mschimmy/Credit_Risk_Analysis/blob/main/Images/split_training_testing_sets.png)

<sub>Splitting the training and testing sets</sub>

After splitting the data into training and testing sets, each sampling method resamples the data according to its specific algorithm. After resampling, the LogisticRegression model is instantiated and used to train the resampled data and make predictions. A balanced accuracy score, confusion matric, and imbalanced classification report is generated for each resampling technique to evaluate the performance of each model.

### Oversampling Models

Once the data has been split into features and the target, and then again into training and testing sets, two oversampling models are built using the RandomOverSampler algorithm and the SMOTE algorithm to resample the data. Once resampled, the LogisticRegression model trains the data.

The [RandomOverSampler](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html) algorithm randomly selects instances of the minority class (high-risk loans) and adds them to the training set until the majority and minority classes are balanced. The calculated balanced accuracy score for the RandomOverSampler algorithm is 0.663. The generated confusion matrix and imbalanced classification report are below and show the precision and recall scores of this model.

![RandomOverSampler confusion matrix](https://github.com/mschimmy/Credit_Risk_Analysis/blob/main/Images/ROS_confusion_matrix.ong.png)

<sub>RandomOverSampler confusion matrix</sub>

![RandomOverSampler imbalanced classification report](https://github.com/mschimmy/Credit_Risk_Analysis/blob/main/Images/ROS_imbalanced_classification_report.png)

<sub>RandomOverSampler imbalanced classification report</sub>


The [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) (synthetic minority oversampling technique) interpolates new values from the minority class and adds them to the training set until the majority and minority classes are balanced. The calculated balanced accuracy score for the SMOTE algorithm is 0.658. The generated confusion matrix and imbalanced classification report are below and show the precision and recall scores of this model.

![SMOTE confusion matrix](https://github.com/mschimmy/Credit_Risk_Analysis/blob/main/Images/SMOTE_confusion_matrix.png)

<sub>SMOTE confusion matrix</sub>

![SMOTE imbalanced classification report](https://github.com/mschimmy/Credit_Risk_Analysis/blob/main/Images/SMOTE_imbalanced_classification_report.png)

<sub>SMOTE imbalanced classification report</sub>


### Undersampling Models

The [ClusterCentroids](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.ClusterCentroids.html) algorithm is used to undersample the data. It does so by identifying clusters of the majority class (low-risk loans),  generating synthetic data points called centroids that are representatives of the clusters, and then removing values from the majority class until the size of the majority class is reduced to that of the minority class. The calculated balanced accuracy score for the ClusterCentroids algorithm is 0.544. The generated confusion matrix and imbalanced classification report are below and show the precision and recall scores of this model.

![ClusterCentroids confusion matrix](https://github.com/mschimmy/Credit_Risk_Analysis/blob/main/Images/ClusterCentroids_confusion_matrix.png)

<sub>ClusterCentroids confusion matrix</sub>

![ClusterCentroids imbalanced classification report](https://github.com/mschimmy/Credit_Risk_Analysis/blob/main/Images/ClusterCentroids_imbalanced_classification_report.png)

<sub>ClusterCentroids imbalanced classification report</sub>


### Combination of Over- and Undersampling Models

The [SMOTEENN](https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTEENN.html) algorithm is used to over- and undersample the data. SMOTEENN combines the SMOTE algorithm with the EEN (edited nearest neighbors) algorithm. SMOTEENN oversamples the minority class with SMOTE and then cleans the resulting data with the ENN undersampling strategy wherein if the two nearest neighbors of a data point belong to two different classes then that data point is dropped. The calculated balanced accuracy score for the SMOTEENN algorithm is 0.679. The generated confusion matrix and imbalanced classification report are below and show the precision and recall scores of this model.

![SMOTEENN confusion matrix](https://github.com/mschimmy/Credit_Risk_Analysis/blob/main/Images/SMOTEENN_confusion_matrix.png)

<sub>SMOTEENN confusion matrix</sub>

![SMOTEENN imbalanced classification report](https://github.com/mschimmy/Credit_Risk_Analysis/blob/main/Images/SMOTEENN_imbalanced_classification_report.png)

<sub>SMOTEENN imbalanced classification report</sub>


### Ensemble Classifier Models

Ensemble classifiers combine multiple models to help improve the accuracy and robustness of the algorithm and decrease the variance of the model. Two ensemble classifiers are used to resample the data: the Balanced Random Forest Classifier and the Easy Ensemble AdaBoost Classifier.

The [Balanced Random Forest Classifier](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html) is used to create 100 estimators that randomly undersampled the data and then trains the data. The calculated balanced accuracy score for the Balanced Random Forest Classifier algorithm is 0.789. The generated confusion matrix and imbalanced classification report are below and show the precision and recall scores of this model.
The Balanced Random Forest Classifier is then used to rank the importance of features in descending order (from most to least important) to determine which feature has the most impact on the predictions. The top five features ranked by importance were "total_rec_prncp", "total_pymnt", "total_pymnt_inv", "total_rec_int", and "last_pymnt_amnt".

![Balanced Random Forest Classifier confusion matrix](https://github.com/mschimmy/Credit_Risk_Analysis/blob/main/Images/BRFC_confusion_matrix.png)

<sub>Balanced Random Forest Classifier confusion matrix</sub>

![Balanced Random Forest Classifier imbalanced classification report](https://github.com/mschimmy/Credit_Risk_Analysis/blob/main/Images/BRFC_imbalanced_classification_report.png)

<sub>Balanced Random Forest Classifier imbalanced classification report</sub>

The [Easy Ensemble AdaBoost Classifier](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html) algorithm is used to create 100 estimators that randomly undersampled the data and then trains the data. The calculated balanced accuracy score for the Easy Ensemble AdaBoost Classifier algorithm is 0.932. The generated confusion matrix and imbalanced classification report are below and show the precision and recall scores of this model.

![Easy Ensemble AdaBoost Classifier confusion matrix](https://github.com/mschimmy/Credit_Risk_Analysis/blob/main/Images/EEAC_confusion_matrix.png)

<sub>Easy Ensemble AdaBoost Classifier confusion matrix</sub>

![Easy Ensemble AdaBoost Classifier imbalanced classification report](https://github.com/mschimmy/Credit_Risk_Analysis/blob/main/Images/EEAC_imbalanced_classification_report.png)

<sub>Easy Ensemble AdaBoost Classifier imbalanced classification report</sub>


## Summary

The purpose of this analysis is to build and perform multiple machine learning algorithms that predict credit risk. The balanced accuracy score as well as the precision and recall scores determine which resampling model best predicts credit risk. Precision is the measure of how reliable a positive classification is; generally, a low precision score is indicative of a large number of false positives. Recall is the ability of the classifier to find all the positive samples; generally, a low recall is indicative of a large number of false negatives. In both cases, a low recall or precision score indicates false classifying a loan application. The best algorithm would be one that has both high precision and recall scores.

Out of the six resampling methods and machine learning models used, the model with the highest precision and accuracy score for correctly identifying high-risk loans is the Easy Ensembler AdaBoost Classifier. This model has a balanced accuracy score of 0.932, a precision score of 0.09, and a recall score of 0.92 for high-risk loans. It is for these reasons that this analysis recommends the client use the Easy Ensembler AdaBoost Classifier to predict credit risk.
