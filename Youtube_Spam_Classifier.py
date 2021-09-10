# -*- coding: utf-8 -*-
"""
COMP237 - NLPProject - Group 3
Youtube Spam Classifier
@authors: 
    Divya Nair - 301169854
    Diego Narvaez - 301082195
    Sreelakshmi Pushpan - 301170860
    Nestor Romero - 301133331
    Jefil Tasna John Mohan - 301149710
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedShuffleSplit
import os
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pandas import DataFrame

### 1. Load data into Pandas Dataframe
data = pd.read_csv('./Youtube03-LMFAO.csv')

### 2. Basic data exploration
"""
print(data.head())
print(data.describe(include='all'))
print(data.CONTENT.describe())
"""

#Dataset with only the selected columns
data_new = data[['CONTENT','CLASS']].copy()

#Comments preprocessing
lemmatizer = WordNetLemmatizer()
corpus = []
for i in range(0,len(data_new)):
    review = re.sub('[^a-zA-Z0-9:)]+', ' ', data_new['CONTENT'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    

corpus = DataFrame(corpus,columns=['comments'])
data_new['CONTENT'] = corpus['comments']

data_new['CONTENT'].eq('').values.any()
data_new.sample(frac=1)

### 3. Data preparation for model building - NLTK
### 4. Output highlights
### 5. Downscale data 
data_X_raw = data_new['CONTENT']
data_y = data_new['CLASS']
count_vectorizer = CountVectorizer()
data_X_vector = count_vectorizer.fit_transform(data_X_raw) # Fit the Data

model_tfidf = TfidfTransformer()
data_X = model_tfidf.fit_transform(data_X_vector)

"""
print(type(data_X_vector))
print(count_vectorizer.vocabulary_)
words_analysis = list(range(len(count_vectorizer.vocabulary_)))
for key in count_vectorizer.vocabulary_:
    value = count_vectorizer.vocabulary_[key]
    words_analysis[value] = key 

print(len(count_vectorizer.get_feature_names()))
print(data_X_vector.shape)
print(data_X_vector[:5])
print(data_X[0:5])

words_analysis[795] #high tfidf
words_analysis[619] #high tfidf
words_analysis[668] #low tfidf
"""

### 7. Split dataset 75-25 

split_data = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
split_data.get_n_splits(data_X, data_y)
for train_index, test_index in split_data.split(data_X, data_y):
        X_train, X_test = data_X[train_index], data_X[test_index]
        y_train, y_test = data_y[train_index], data_y[test_index]
        
### 8. Naive Bayes Classifier
classifier = GaussianNB()
classifier.fit(X_train.toarray(),y_train)
classifier.score(X_test.toarray(),y_test)
y_pred = classifier.predict(X_test.toarray())

### 9. Cross validate model 5-fold
### 10. Testing results, confusion matrix and accuracy
score = cross_val_score(classifier, X_train.toarray(), y_train, scoring='accuracy', cv=5);
print(score);
print('Mean of Score: ', score.mean())

results = pd.DataFrame(data = {'comments' : X_test, 'result' : y_pred, 'expected': y_test })

accuracy = accuracy_score(y_test,y_pred)
print('\n','Accuracy')
print(accuracy, '\n\n')

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred), '\n')

print("Classification Report:")
classification_report=classification_report(y_test,y_pred)
print(classification_report)

### 11. Test the model
input_data = [
    'wierd but funnyï»¿¿', 
    'Party Rock....lol...who wants to shuffle!!!ï»¿',
    'wow!!!!!! increible song!!!!!!!!!ï»¿',
    'Best song ever!!!!ï»¿',
    'give it a likeï»¿',
    'Check out this video on YouTube:ï»¿',
    'One of my favorite videos',
    'Divya and Jefil could totally make this dance',
    'Sreelakshmi great recommendation!!',
    'Diego this a new recommendation for your playlist',
    'Nestor was this song popular in Colombia?'
]

# Transform input data using count vectorizer
input_tc = count_vectorizer.transform(input_data)
type(input_tc)

# Transform vectorized data using tfidf transformer
input_tfidf = model_tfidf.transform(input_tc)
type(input_tfidf)
# Predict the output categories
predictions = classifier.predict(input_tfidf.toarray())

# Print the outputs
for sent, category in zip(input_data, predictions):
    if category == 0:
        label = 'Ham'
    else:
        label = 'Spam'
    print('\nInput:', sent, '\nPredicted category:', \
            category,'-', label)
