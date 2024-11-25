import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm, datasets
from sklearn.linear_model import LinearRegression 
import sklearn.metrics as sm
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
import tensorflow as tf

def minmaxscaler(data):
    numerator=data-np.mean(data,0)
    denominator=np.max(data,0)-np.min(data,0)
    return numerator/(denominator+1e-7)
t=pd.read_csv('train_numb.csv')
x=t[['Age','Gender','Education_Status','Employment_Status','Working_Week (Yearly)','Industry_Status','Occupation_Status','Race','Hispanic_Origin','Martial_Status'
,'Citizenship','Birth_Country','Tax_Status','Income_Status']]
y=t['Income']
x_data=minmaxscaler(x)


test=pd.read_csv('test_numb.csv')
a=test[['Age','Gender','Education_Status','Employment_Status','Working_Week (Yearly)','Industry_Status','Occupation_Status','Race','Hispanic_Origin','Martial_Status'
,'Citizenship','Birth_Country','Tax_Status','Income_Status']]


test_data=minmaxscaler(a)
