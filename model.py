from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import pickle

framingham = pd.read_csv('/Users/ujjawalbabu/Downloads/framingham.csv')
framingham.fillna(framingham.median(), inplace=True)
X_framingham = framingham.drop('TenYearCHD', axis=1)
y_framingham = framingham['TenYearCHD']

X_train, X_test, y_train, y_test = train_test_split(X_framingham, y_framingham, random_state=1 , train_size=0.75)
model = GaussianNB()                  # instantiation of Naive Bayes model
model.fit(X_train.values, y_train.values)           # fit data to the model

pickle.dump(model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

print(model.predict([[0,634341,3.0,1,30.0,0.0,0,1,0,225.0,1534340.0,95.0,2434348.58,65.0,103.0]]))