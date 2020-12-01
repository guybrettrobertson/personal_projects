# Credit card fraud detection

The purpose of this project is to build a machine learning model capable of determining whether a credit card transaction is fraudulent or not.

A dataset of c. 284,000 credit card transactions has been obtained from [Kaggle.com](https://www.kaggle.com/mlg-ulb/creditcardfraud). I first pre-processed the data and reduced the dimensionality by retaining a subset of the principal components of the data. I then visualise the correlations present in the data and use histograms to understand the distributions of fraudulent and non-fraudulent transactions for each principal component.

I compared a number of candidate models including logistic regression, a decision tree, and a neural network. The best candidate model was the neural network and after further tuning the hyperparameters of this model I achieved an ROC AUC score of 99.7%.

This project uses Python and a number of packages including NumPy and Pandas for data manipulation, Matplotlib and Seaborn for data visualisation, and scikit-learn for creating, fit, and evaluating the machine learnings models.
