# Model Card

## Model Details
Random Forest Classifier is used with default hyperparameters setting from scikit-learn.

## Intended Use
This model is to predict whether a person's annual salary will be higher or lower than $50K based on PI.

## Data
Data is publicly available through Census Bureau. Data has 32561 records with 15 features. Data split into 80% of training and 20% of test dataset. To process the data, One Hot Encoding is used for categorical features. Data has no null values, and thus imputation was not necessary.

## Metrics
Three metrics were used to evaluate model performance.
Precision: 0.9556425309849967
Recall: 0.9325270528325907
Fbeta: 0.9439432989690721

These metrics are measured for each group of categorical features. Please refer to slice_output.txt in outputs folder.

## Ethical Considerations
This data contains PI(Personal Information), which can explicitly and implicitly affect the prediction. This is why data slicing is important to ensure the similar prediction across groups in a feature. 

## Caveats and Recommendations
This is the baseline Random Forest Classifier and thus is not tuned for hyperparameter. For the better performance, trying other algorithms with hyperparameter optimization can provide better outcome.