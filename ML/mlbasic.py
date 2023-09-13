#%%
import sqlite3 #sqlite3 데이터베이스
import re #정규식
import numpy as np #숫자 
import pandas as pd #데이터 처리  
import matplotlib.pyplot as plt #그래프 
import matplotlib 
import seaborn as sns #그래프 고도화
#scikit-learn 
#%%
from sklearn import datasets as data #scikit-learn의 데이터셋들
#dir(data)
iris = data.load_iris()
#%%
irdata = iris.data #데이터
irtgt = iris.target #라벨링
feature = iris.feature_names #데이터 컬럼명
tgtname = iris.target_names
#%%
feature = ['sl','sw','pl','pw']
# %%
df=pd.DataFrame(irdata,columns=feature)
df
# %%
df.plot() # 기초 그래프
#%%
df.plot(style='.')
# %%
df.describe() # 기초통계 요약
#%%
df.info() # 데이터 타입 요약
# %%
plt.hist(irtgt) # 카테고리별 갯수 히스토그램
# %%
df['tgt']=irtgt
df
# %% 
# seaborn의 pairplot 관계그래프 라이브 
# corner=True 반만 보기
sns.pairplot(df,hue='tgt',corner=True,palette='husl') # 3개의 그룹이 분류되는가 등등...
#%%
plt.plot(irtgt) #만약 이 그림에서 70%
# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split #시험문제 분리하기
from sklearn.metrics import accuracy_score # 정확도 계산하기
X_train,X_test,Y_train,Y_test=train_test_split(irdata,irtgt,test_size=0.3,shuffle=True,random_state=1) #shuffle => 문제 섞어라, random_state => 
print(X_train.shape,X_test.shape)#%%train과 test 잘 분리되었는지 확인
print(Y_train.shape,Y_test.shape)#%%

#%% 하이퍼 파이미터 튜닝 KNN
for i in range(3,20,2):
    print('knn:',i)
    knn3 = KNeighborsClassifier(n_neighbors=i) # 모델지정 :근처 3개를 기준으로 분류
    knn3.fit(X_train,Y_train) # 학습시키기
    pred = knn3.predict(X_test) # 시험보기
    print(pred) # 시험본 답 출력
    print(Y_test) # 실제 답 출력
    acc = accuracy_score(pred,Y_test) # 답지랑 비교해서 맞는지 확인
    print('점수[',i,'] :',acc)
#%% SVM
from sklearn.svm import SVC
for i in range(1,10):
    svc=SVC(C=i)
    svc.fit(X_train,Y_train)
    pred=svc.predict(X_test)
    print(pred)
    print(Y_test)
    acc=accuracy_score(pred,Y_test)
    print('SVM [',i,'] acc:',acc)
# %% 디시젼트리
from sklearn.tree import DecisionTreeClassifier as DT
for j in range(2,10):
    for i in range(2,10):
        dt=DT(max_depth=i,min_samples_leaf=j)
        dt.fit(X_train,Y_train)
        pred = dt.predict(X_test)
        print(pred)
        print(Y_test)
        acc=accuracy_score(pred,Y_test)
        print('DT[',i,',',j,'] acc:',acc)