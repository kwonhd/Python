#%%
import sqlite3 #sqlite3 데이터베이스
import re #정규식
import numpy as np #숫자 
import pandas as pd #데이터 처리  
import matplotlib.pyplot as plt #그래프 
import matplotlib 
import seaborn as sns #그래프 고도화
# %% 형태소 분석기
from konlpy.tag import Kkma
# %%
kkma=Kkma()
# %%
res = kkma.pos('안녕하세요 그런데 여러분 만나서 반갑습니다.')
res
# %%
def getPOS(txt='안녕하세요 그런데 여러분 만나서 반갑습니다.'):
    res=kkma.pos(txt)
    reqPos=['NNG','NNP','NP','VV','VA','VCN','JC','MAC','EFA','EFQ','EFO','EFA','EFI','EFR']#,'VCP','EFN'
    wset = []
    for r in res:
        if(r[1] in reqPos):
            wset.append(r[0])
            #print(r)
    return(' '.join(wset))
getPOS()
#%%
fname = '../ML/src/현진건-운수_좋은_날+B3356-개벽.txt'
with open(fname,encoding='utf-8') as f:
    r=f.readlines()
print(r)
# %%
lucky=''.join(r) # 한줄로 만들기 / \n과 \n\n의 차이 확인
lucky
# %%
lucky = lucky.replace('\n\n','{nn}')
lucky = lucky.replace('\n','')
lucky = lucky.replace('{nn}','.')

print(lucky)
# %%
luckys = lucky.split('.') # split문장 내장함수
luckys[:10]
# %%
copus=[]
for luck in luckys:
    ltxt=getPOS(luck)
    copus.append(ltxt)
copus[:10]
# %%
#t =  '새침하게 흐린 품이 눈이 올 듯하더니 눈은 아니 오고 얼다가 만 비가 추적추적 내리는 날이었다'
#getPOS(t)

# %%
# 단어 행렬 CBOW
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
#CountVectorizer =>숫자세서 벡터화 / TfidfVectorizer=>많이 나오는 단어 없애기
cvect = CountVectorizer()
cvfit=cvect.fit_transform(copus)
cvtable = cvfit.toarray()
print(cvtable.shape)
print(cvect.vocabulary_) # 단어별 인덱스
# %%
plt.imshow(cvtable[:100,:100])

# %%
#빈도 - 역빈도 TFIDF
tvect = TfidfVectorizer() # 특징추출
tvfit=tvect.fit_transform(copus)
tvtable = tvfit.toarray()
print(tvtable.shape)
print(tvect.vocabulary_) # 단어별 인덱스

# %%
plt.imshow(cvtable[:100,:100])
# %%
cdf = pd.DataFrame(cvtable[:100,:100])
cdf
#%%
tdf = pd.DataFrame(tvtable[:100,:100])
tdf
# %%
cbdf = pd.DataFrame(cvtable)
print(cvect.vocabulary_)
voca = cvect.vocabulary_
vosort={k:v for k, v in sorted(voca.items(),key=lambda item : item[1])} # 람다식
cols = vosort.keys()
cbdf = pd.DataFrame(cvtable,columns=cols)
cbdf
# %% 유사도 찾기
from sklearn.metrics.pairwise import cosine_similarity,manhattan_distances #코사인, 맨하탄
#%%
def getCosim(v1,v2): # 운수좋은날 vector로 
    cosim = cosine_similarity(v1,v2)
    return (cosim)
v1=cbdf.values
csimv = getCosim(v1,v1) #본인과 본인 넣기, 같은 글임
print(csimv.shape)
csimv[2]

# %%
import numpy as np
simtxt = np.where(csimv[2]>0.1) #() 변경 시 단어 변경되어 유사도 탐색 및 비슷한 문장 추천
#%%
for sim in simtxt[0]:
    print(luckys[sim])
    print('-'*30)
    print(np.round(csimv[2][sim],2),',',luckys[sim]) # 유사도 몇인지 파악
    print('-'*30)
#%%
