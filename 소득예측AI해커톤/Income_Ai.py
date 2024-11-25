import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn import linear_model
test=pd.read_csv('train.csv')
names=[]
for i in test.columns:
  try:
      int(test.iloc[1][i])
  except:
    if i=="ID":
      pass
    else:
      uni=test[i].unique()
      for a in uni:
        names.append(a)
dicts=[]
i=0
while i <= len(names):
  dicts.append(i)
  i+=1
dict_list=dict(zip(names,dicts))

# 숫자로 바꾸기
f = open('train.csv', "r")
reader = csv.reader(f)
result=[]
for row in reader:
    data_numb=[]
    i=0
    for b in row:
        if type(b) is str:
            c=True
            for a in dict_list: 
                if a==b:
                    temp=b.replace(b,str(int(dict_list[a])))
                    data_numb.append(temp)
                    c=False
            if c:
               data_numb.append(b)
    result.append(data_numb)
with open('train_numb.csv','w', newline='') as file:
   writer=csv.writer(file)
   writer.writerows(result)

#test 데이터 만들기
f = open('test.csv', "r")
reader = csv.reader(f)
result=[]
for row in reader:
    data_numb=[]
    i=0
    for b in row:
        if type(b) is str:
            c=True
            for a in dict_list: 
                if a==b:
                    temp=b.replace(b,str(int(dict_list[a])))
                    data_numb.append(temp)
                    c=False
            if c:
               data_numb.append(b)
    result.append(data_numb)
print(result)
with open('test_numb.csv','w', newline='') as file:
   writer=csv.writer(file)
   writer.writerows(result)
# 분석  
# t=pd.read_csv('train_numb.csv')
# x=t[['Age','Gender','Education_Status','Employment_Status','Working_Week (Yearly)','Industry_Status','Occupation_Status','Race','Hispanic_Origin','Martial_Status'
# ,'Citizenship','Birth_Country','Tax_Status','Income_Status']]
# y=t['Income']
# regr=linear_model.LinearRegression()
# regr.fit(x,y)