# consumer_complaints_interpretation

Kaggle.com has provided a large dataset of complaints written by consumers, along with labels for the products and issues (a.k.a. categories) related to the complaint. In this project, I have implemented natural language processing techniques including tokenization, the removal of stop words, stemming and lemmatization, and tf-idf and count vectorization to pre-process the data. I then used scikit-learn classification models, particularly a multinomial naive Bayes classifier, in order to predict the complaint issue cited by the consumer for each complaint.

This is a difficult modelling challenge because there are 90 different classes of issue that each complaint could relate to, and the complaints data is highly dirty, containing a large number of spelling and typographical mistakes. Notwithstanding these limitations, I was able to develop a model that could classify 53% of test complaints accurately.
