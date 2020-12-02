# Audio data classification

The Spotify API has been used to produce a dataset of c.5,000 songs. Each song has a set of quantifiable audio features such as acousticness, energy, and tempo. Each song also has a genre (either hip-hop or rock for this dataset). In this project, I have implemented machine learning models to classify the songs by genre using the audio-features.

I normalised the data for principal component analysis, which I implemented to reduce the dimensionality of the data. I also balanced the data because there was a preponderance of rock songs. I implemented a decision tree classifier and a logistic regression model, using cross-validation to determine the best model performance. With the logistic regression classifier, I achieved a cross-validation score of 78%.

This project uses Python and a number of packages including NumPy and Pandas for data manipulation, Matplotlib and Seaborn for data visualisation, and scikit-learn for creating, fit, and evaluating the machine learnings models.
