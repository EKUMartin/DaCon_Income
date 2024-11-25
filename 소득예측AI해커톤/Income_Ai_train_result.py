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

t=pd.read_csv('train_numb.csv')
x=t[['Age','Gender','Education_Status','Employment_Status','Working_Week (Yearly)','Industry_Status','Occupation_Status','Race','Hispanic_Origin','Martial_Status'
,'Citizenship','Birth_Country','Tax_Status','Income_Status']]
y=t['Income']

# param_grid = {'criterion':['mse','friedman_mse','mae'], 'max_depth':[None,2,3,4,5,6], 'max_leaf_nodes':[None,2,3,4,5,6,7], 'min_samples_split':[2,3,4,5,6], 'min_samples_leaf':[1,2,3], 'max_features':[None,'sqrt','log2',3,4,5]}

model=DecisionTreeRegressor()
print("A")
# grid = GridSearchCV(model, param_grid=param_grid)
# grid.fit(x,y)
model.fit(x,y)
# print(grid.best_score_)
# print(grid.best_params_)
test=pd.read_csv('test_numb.csv')
a=test[['Age','Gender','Education_Status','Employment_Status','Working_Week (Yearly)','Industry_Status','Occupation_Status','Race','Hispanic_Origin','Martial_Status'
,'Citizenship','Birth_Country','Tax_Status','Income_Status']]

pred=model.predict(a)

print(model.score(x,y))
# id=[]
# for a in test['ID']:
#    id.append(a)
# i=0
# c="ID"
# d="Income"
# head=[]
# result=[]
# head.append(c)
# head.append(d)
# result.append(head)
# while i< len(id):
#    b=[]
#    b.append(id[i])
#    b.append(pred[i])
#    result.append(b)
#    i+=1
# with open('submission.csv','w', newline='') as file:
#    writer=csv.writer(file)
#    writer.writerows(result)

# 히트맵
# plt.figure(figsize=(10,8))
# sns.heatmap(t.corr(), 
#            annot=True, cmap='Reds', vmin=-1,vmax=1) # annot : 주석
# plt.show()
# 결과
# plt.figure(figsize=(12,8))
# plt.scatter(test['ID'],pred,color='red')
# plt.scatter(t['ID'],y)
# plt.show()