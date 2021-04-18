# -*- coding: utf-8 -*-
# DataScience Project TCNER - Group -20 - Émile Heijs & Hrishikesh Mane

# imports
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer as WNL
import re
import string
import time
from sklearn.feature_extraction.text import TfidfVectorizer

directory = 'C:/Users/emile/Documents/Master 2A/Data Science/Project/'

dblp_train = pd.read_csv(directory + 'DBLPTrainset.txt', sep='\t', header=None, usecols=[1,2], names=["Conference", "Article Title"])
dblp_test = pd.read_csv(directory + 'DBLPTestset.txt', sep='\t', header=None, usecols=[1], names=["Article Title"])
ground_truth = pd.read_csv(directory + 'DBLPTestGroundTruth.txt', sep='\t', header=None, usecols=[1], names=["Conference"])

#%%

from sklearn.decomposition import TruncatedSVD
from sklearn import svm
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

#function to preprocess the data 
def preprocess(text):
    
    lematizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    htmltags     = re.compile('<.*?>|(,*?)')
    cleaned_word = re.sub(htmltags,'',text)
    #cleaned_word = text.lower()
    cleaned_word = text.replace('\n','')
    tokenized = word_tokenize(cleaned_word)
    stop_words = set(stopwords.words('english'))
    tokenized = [word.lower() for word in tokenized if word not in stop_words]
    tokenized = [lematizer.lemmatize(word) for word in tokenized]
    #tokenized = [stemmer.stem(word) for word in tokenized]
    tokenized = [word for word in tokenized if len(word)>2]
    tokenized = [word for word in tokenized if word.isalpha()]
    cleanedText = " ".join(tokenized)
    return cleanedText

#feature selection using tfidf 
def tfidf_feature(text):
    tfidf = TfidfVectorizer(sublinear_tf=True,min_df=4,max_df=0.9,norm='l2',ngram_range=(1,2))
    feature = tfidf.fit_transform(text)
    return feature

def featuresByLSA(features,ncomponents=200):
    svd = TruncatedSVD(n_components=ncomponents)
    normalizer =  Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    dtm_lsa = lsa.fit_transform(features)
    return dtm_lsa

#creating a copy of train and test data frame
train_df_copy = dblp_train.copy()
train_df_copy['Article Title'] = train_df_copy['Article Title'].apply(preprocess)
encoder = LabelEncoder()
train_df_copy['label'] = encoder.fit_transform(train_df_copy['Conference']) 

# extracting features
text = train_df_copy["Article Title"].astype('str')
features_tfidf = tfidf_feature(text)
lsa = featuresByLSA(features_tfidf)

#%%

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline

dimreduction = True

if dimreduction:
    ncomponents = 1200
    steps = [('svd', TruncatedSVD(n_components=ncomponents)), ('m', DecisionTreeClassifier())]
    model_dt = Pipeline(steps=steps)
    steps = [('svd', TruncatedSVD(n_components=ncomponents)), ('m', LinearSVC())]
    model_lsv = Pipeline(steps=steps)
    steps = [('svd', TruncatedSVD(n_components=ncomponents)), ('m', LogisticRegression(max_iter=1000))]
    model_lr = Pipeline(steps=steps)
    steps = [('svd', TruncatedSVD(n_components=ncomponents)), ('m', RandomForestClassifier())]
    model_rf = Pipeline(steps=steps)
    steps = [('svd', TruncatedSVD(n_components=ncomponents)), ('m', xgb.XGBClassifier())]
    model_xgb = Pipeline(steps=steps)
else:
    model_dt = DecisionTreeClassifier()
    model_lsv = LinearSVC()
    model_lr = LogisticRegression(max_iter=1000)
    model_rf = RandomForestClassifier()
    model_xgb = xgb.XGBClassifier()

# naive bayes
model_nb = MultinomialNB()
cv_scores = cross_val_score(model_nb, features_tfidf, train_df_copy["Conference"], cv=5)
print(" ")
print("Cross validation scores NB:", cv_scores)

# decision tree
cv_scores = cross_val_score(model_dt, features_tfidf, train_df_copy["Conference"], cv=5)
print(" ")
print("Cross validation scores DT:", cv_scores)

# support vector machine
cv_scores = cross_val_score(model_lsv, features_tfidf, train_df_copy["Conference"], cv=5)
print(" ")
print("Cross validation scores LSV:", cv_scores)

# logistic regression
cv_scores = cross_val_score(model_lr, features_tfidf, train_df_copy["Conference"], cv=5)
print(" ")
print("Cross validation scores LR:", cv_scores)

# random forest
cv_scores = cross_val_score(model_rf, features_tfidf, train_df_copy["Conference"], cv=5)
print(" ")
print("Cross validation scores RF:", cv_scores)

# xg boost
cv_scores = cross_val_score(model_xgb, features_tfidf, train_df_copy["label"], cv=5)
print(" ")
print("Cross validation scores XGB:", cv_scores)

#%%
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score

x_train, x_test, y_train, y_test = train_test_split(features_tfidf, train_df_copy["label"], test_size=0.20, random_state=1)

model_nb = MultinomialNB()

dimreduction = False

if dimreduction:
    ncomponents = 1200
    steps = [('svd', TruncatedSVD(n_components=ncomponents)), ('m', DecisionTreeClassifier())]
    model_dt = Pipeline(steps=steps)
    steps = [('svd', TruncatedSVD(n_components=ncomponents)), ('m', LinearSVC())]
    model_lsv = Pipeline(steps=steps)
    steps = [('svd', TruncatedSVD(n_components=ncomponents)), ('m', LogisticRegression(max_iter=1000))]
    model_lr = Pipeline(steps=steps)
    steps = [('svd', TruncatedSVD(n_components=ncomponents)), ('m', RandomForestClassifier())]
    model_rf = Pipeline(steps=steps)
    steps = [('svd', TruncatedSVD(n_components=ncomponents)), ('m', xgb.XGBClassifier())]
    model_xgb = Pipeline(steps=steps)
else:
    model_dt = DecisionTreeClassifier()
    model_lsv = LinearSVC()
    model_lr = LogisticRegression(max_iter=1000)
    model_rf = RandomForestClassifier()
    model_xgb = xgb.XGBClassifier()
    
    
# fit models
model_nb.fit(x_train, y_train)
model_dt.fit(x_train, y_train)
model_lsv.fit(x_train, y_train)
model_lr.fit(x_train, y_train)
model_rf.fit(x_train, y_train)
model_xgb.fit(x_train, y_train)

# predict on train subset
prediction_nb = model_nb.predict(x_test)
prediction_dt = model_dt.predict(x_test)
prediction_lsv = model_lsv.predict(x_test)
prediction_lr = model_lr.predict(x_test)
prediction_rf = model_rf.predict(x_test)
prediction_xgb = model_xgb.predict(x_test)

# calculate statistics
cfm_nb = confusion_matrix(y_test, prediction_nb)
report_nb = classification_report(y_test, prediction_nb)
mapc_nb = precision_score(y_test, prediction_nb, average='micro')
cfm_dt = confusion_matrix(y_test, prediction_dt)
report_dt = classification_report(y_test, prediction_dt)
cfm_lsv = confusion_matrix(y_test, prediction_lsv)
report_lsv = classification_report(y_test, prediction_lsv)
mapc_lsv = precision_score(y_test, prediction_lsv, average='micro')
cfm_lr = confusion_matrix(y_test, prediction_lr)
report_lr = classification_report(y_test, prediction_lr)
mapc_lr = precision_score(y_test, prediction_lr, average='micro')
cfm_rf = confusion_matrix(y_test, prediction_rf)
report_rf = classification_report(y_test, prediction_rf)
cfm_xgb = confusion_matrix(y_test, prediction_xgb)
report_xgb = classification_report(y_test, prediction_xgb)

print(" ")
print("Naive Bayes:")
print(cfm_nb)
print(" ")
print(report_nb)
print(" ")
print("Micro-average Precision", mapc_nb)
print(" ")
print("Decision Tree:")
print(cfm_dt)
print(" ")
print(report_dt)
print(" ")
print("Linear Support Vector:")
print(cfm_lsv)
print(" ")
print(report_lsv)
print(" ")
print("Micro-average Precision", mapc_lsv)
print(" ")
print("Logistic Regression:")
print(cfm_lr)
print(" ")
print(report_lr)
print(" ")
print("Micro-average Precision", mapc_lr)
print(" ")
print("Random Forest:")
print(cfm_rf)
print(" ")
print(report_rf)
print(" ")
print("XG Boost:")
print(cfm_xgb)
print(" ")
print(report_xgb)

#%%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# preprocess test dataset and extract features
test_df_copy = dblp_test.copy()
test_df_copy["Article Title"] = test_df_copy["Article Title"].apply(preprocess)
test_text = test_df_copy["Article Title"].astype('str')
test_features_tfidf = tfidf_feature(test_text)

# vectorize features
vectorizer_train = CountVectorizer()
features_tfidf = vectorizer_train.fit_transform(text)
vectorizer_test = CountVectorizer(vocabulary=vectorizer_train.vocabulary_)
test_features_tfidf = vectorizer_test.fit_transform(test_text)

model_nb = MultinomialNB()

dimreduction = True

if dimreduction:
    # set models using dimension reduction
    ncomponents = 1200
    steps = [('svd', TruncatedSVD(n_components=ncomponents)), ('m', DecisionTreeClassifier())]
    model_dt = Pipeline(steps=steps)
    steps = [('svd', TruncatedSVD(n_components=ncomponents)), ('m', LinearSVC())]
    model_lsv = Pipeline(steps=steps)
    steps = [('svd', TruncatedSVD(n_components=ncomponents)), ('m', LogisticRegression(max_iter=1000))]
    model_lr = Pipeline(steps=steps)
    steps = [('svd', TruncatedSVD(n_components=ncomponents)), ('m', RandomForestClassifier())]
    model_rf = Pipeline(steps=steps)
    #steps = [('svd', TruncatedSVD(n_components=ncomponents)), ('m', xgb.XGBClassifier())]
    #model_xgb = Pipeline(steps=steps)
else:
    model_dt = DecisionTreeClassifier()
    model_lsv = LinearSVC()
    model_lr = LogisticRegression(max_iter=1000)
    model_rf = RandomForestClassifier()
    model_xgb = xgb.XGBClassifier()


# train models again on full train set
model_nb.fit(features_tfidf, train_df_copy["Conference"])
model_dt.fit(features_tfidf, train_df_copy["Conference"])
model_lsv.fit(features_tfidf, train_df_copy["Conference"])
model_lr.fit(features_tfidf, train_df_copy["Conference"])
model_rf.fit(features_tfidf, train_df_copy["Conference"])
#model_xgb.fit(features_tfidf, train_df_copy["Conference"])

# predict test set
gt_prediction_nb = model_nb.predict(test_features_tfidf)
gt_prediction_dt = model_dt.predict(test_features_tfidf)
gt_prediction_lsv = model_lsv.predict(test_features_tfidf)
gt_prediction_lr = model_lr.predict(test_features_tfidf)
gt_prediction_rf = model_rf.predict(test_features_tfidf)
#gt_prediction_xgb = model_xgb.predict(test_features_tfidf)

# get statistics
cfm_nb = confusion_matrix(ground_truth, gt_prediction_nb)
report_nb = classification_report(ground_truth, gt_prediction_nb)
mapc_nb = precision_score(ground_truth, gt_prediction_nb, average='micro')
cfm_dt = confusion_matrix(ground_truth, gt_prediction_dt)
report_dt = classification_report(ground_truth, gt_prediction_dt)
cfm_lsv = confusion_matrix(ground_truth, gt_prediction_lsv)
report_lsv = classification_report(ground_truth, gt_prediction_lsv)
mapc_lsv = precision_score(ground_truth, gt_prediction_lsv, average='micro')
cfm_lr = confusion_matrix(ground_truth, gt_prediction_lr)
report_lr = classification_report(ground_truth, gt_prediction_lr)
mapc_lr = precision_score(ground_truth, gt_prediction_lr, average='micro')
cfm_rf = confusion_matrix(ground_truth, gt_prediction_rf)
report_rf = classification_report(ground_truth, gt_prediction_rf)
#cfm_xgb = confusion_matrix(ground_truth, gt_prediction_xgb)
#report_xgb = classification_report(ground_truth, gt_prediction_xgb)

print(" ")
print("Multinomial Naive Bayes - GT:")
print(cfm_nb)
print(" ")
print(report_nb)
print(" ")
print("Micro-average Precision", mapc_nb)
print(" ")
print("Decision Tree - GT:")
print(cfm_dt)
print(" ")
print(report_dt)
print(" ")
print("Linear Support Vector - GT:")
print(cfm_lsv)
print(" ")
print(report_lsv)
print(" ")
print("Micro-average Precision", mapc_lsv)
print("Logistic Regression - GT:")
print(cfm_lr)
print(" ")
print(report_lr)
print(" ")
print("Micro-average Precision", mapc_lr)
print("Random Forrest - GT:")
print(cfm_rf)
print(" ")
print(report_rf)
print("XG Boost - GT:")
#print(cfm_xgb)
print(" ")
#print(report_xgb)


#%% parameter optimization

from sklearn.model_selection import RandomizedSearchCV

classifier_nb = MultinomialNB()
classifier_dt = DecisionTreeClassifier()

parameters_nb = classifier_nb.get_params()
parameters_dt = classifier_dt.get_params()

params_nb = {
    "alpha" : [0.25,0.5,1,1.25, 1.5, 1.75, 2],
    "fit_prior" : [True, False]
}

params_dt = {
    "ccp_alpha" : [0.0, 0.005, 0.01, 0.015, 0.02, 0.025]    
}

random_search_nb = RandomizedSearchCV(classifier_nb, param_distributions=params_nb, n_iter=5, cv=5)
random_search_dt = RandomizedSearchCV(classifier_dt, param_distributions=params_dt, n_iter=5, cv=5)

random_search_nb.fit(features_tfidf, train_df_copy["Conference"])
random_search_dt.fit(features_tfidf, train_df_copy["Conference"])

print(random_search_nb.best_estimator_)
print(random_search_dt.best_estimator_)
