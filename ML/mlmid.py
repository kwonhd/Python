#%%
#%%
import sqlite3 #sqlite3 데이터베이스
import re #정규식
import numpy as np #숫자 
import pandas as pd #데이터 처리  
import matplotlib.pyplot as plt #그래프 
import matplotlib 
import seaborn as sns #그래프 고도화

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=load_breast_cancer()
X=data['data']
Y=data['target']
fname=data['feature_names']
tname=data['target_names']
#print(data['DESCR']) # 데이터 설명
df=pd.DataFrame(X,columns=fname)
df

#%%
#EDA
#기초통계량
df.describe()
#%%
df.info()
#%%
#기초시각화
plt.hist(Y)
Y
#%%
plt.plot(df['mean radius'],'.')
sns.scatterplot(df.iloc[:,:3])
#%%
tdf=df.copy()
tdf['tgt']=Y
sns.pairplot(tdf.iloc[:,-5:],hue='tgt')
#---------------------------------------------------------------------#
#%% 데이터 전처리
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,shuffle=True,random_state=1,stratify=Y)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
plt.hist(Y_train)
plt.hist(Y_test)

# %% 앙상블 모델 Random Forest
from sklearn.ensemble import RandomForestClassifier as RF
#%%하이퍼 파라미터 튜닝
def makeRF(i,j):
    rf=RF(max_depth=j,max_leaf_nodes=i)
    rf.fit(X_train,Y_train)
    pred = rf.predict(X_test)
    acc=accuracy_score(pred,Y_test)
    print('RF[',i,',',j,'] acc:',acc)
    return acc
accs=[]
beforeACC=0
bestACC=[]
for i in range(2,10):
    for j in range(2,10):
        acc=makeRF(i,j)
        if(acc>beforeACC):
            bestACC=[i,j,acc]
        beforeACC=acc
        accs.append(acc)        
#%%
print(bestACC)
plt.plot(accs)

# %% 결정트리
from sklearn.tree import DecisionTreeClassifier as DT
#%%
#DT 하이퍼 파라미터 튜닝
def makeDT(i,j):
    rf=DT(max_depth=j,max_leaf_nodes=i)
    rf.fit(X_train,Y_train)
    pred = rf.predict(X_test)
    acc=accuracy_score(pred,Y_test)
    print('DT[',i,',',j,'] acc:',acc)
    return acc
accs=[]
beforeACC=0
bestACC=[]
for i in range(2,10):
    for j in range(2,10):
        acc=makeDT(i,j)
        if(acc>beforeACC):
            bestACC=[i,j,acc]
        beforeACC=acc
        accs.append(acc)        
#%%
print(bestACC)
plt.plot(accs)
# %%
from sklearn.ensemble import GradientBoostingClassifier as GB
#그래디언트 부스팅 - 무거움

def makeGB(i,j):
    rf=GB(min_samples_split=j,n_estimators=i*50) # n_estimators => 생성할 트리 갯수
    rf.fit(X_train,Y_train)
    pred = rf.predict(X_test)
    acc=accuracy_score(pred,Y_test)
    print('GB[',i,',',j,'] acc:',acc)
    return acc
accs=[]
beforeACC=0
bestACC=[]
for i in range(1,10):
    for j in range(2,10):
        acc=makeGB(i,j)
        if(acc>beforeACC):
            bestACC=[i,j,acc]
        beforeACC=acc
        accs.append(acc)        
#%%
print(bestACC)
plt.plot(accs)

# %%
from sklearn.ensemble import AdaBoostClassifier as AB
# 아다 부스팅 - GradientBoostingClassifier + 모멘텀

def makeAB(i):
    rf=AB(n_estimators=i*20) # n_estimators => 생성할 트리 갯수
    rf.fit(X_train,Y_train)
    pred = rf.predict(X_test)
    acc=accuracy_score(pred,Y_test)
    print('AB[',i,',',j,'] acc:',acc)
    return acc
accs=[]
beforeACC=0
bestACC=[]
for i in range(1,10):
    acc=makeAB(i)
    if(acc>beforeACC):
        bestACC=[i,acc]
    beforeACC=acc
    accs.append(acc)        
#%%
print(bestACC)
plt.plot(accs)
# %%
