#%%
print('hello')

# %%
a=3
b=5
c=a+b
print(c)

# %%
print('나머지연산:',b%a)
'몫연산:',b//a
# %%
#데이터 셋: array
    #리스트 : [] - 일반배열
    #튜플 : () - 수정 불가능한 리스트
    #딕셔너리{키:벨류} - 연관배열 : 객체의 개념
    #셋 : set() 집합
    
li=[1,2,'3',[4,5]] #리스트 인덱스 숫자로 접근
li[0] = 0
print('0치환 이후:',li)
#%%
tu=(1,2,'3',[4,5]) #튜플 수정 불가능
tu[0] = 0
print('0치환 이후:',tu)

# %%
di={'이름:':'홍길동',1:19,2:[1,2,3]}
print(di[2])
# %%
# {}블럭이 없어서 들여쓰기가 블럭으로 인식
for i in li:
    print(i)
# %%
for t in tu:
    print(t)
# %%
for d in di:
    print(d)
    
print('#'*30)

for k,v in di.items():
    print(k,v)
# %%
def make99():
    for i in range(1,10,1):
        print('')
        for j in range(1,10):
            ans=i*j
            if((i*j)%2==1):
                print(ans,'*',end='\t')
            else:
                print(ans,end='\t') #end='' = \n없음 , end=',' = ,로 나눔
            
            
# %%
i=5
while(i<10):
    i+=1
    print(i)
# %%
#def 함수명():
def add(a,b):
    return a+b
add(2,3)

# %%
#클래스 선언
class person():
    def __init__(self,name,age):
        self.name=name
        self.age=str(age)
    def sayHello(self): # 클래스의 메서드이다 self
        print('Hello'+self.name)
        print('I"m'+self.age+'year old')
anna=person('anna',19)
anna.sayHello()
# %%
