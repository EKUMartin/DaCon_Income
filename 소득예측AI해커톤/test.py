import pandas as pd
test=pd.read_csv('test_numb.csv')
t=pd.read_csv('train_numb.csv')
for a in t:
    print(a)