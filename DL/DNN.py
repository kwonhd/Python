#%% tensorflow
# * 케라스 PYTHON < 파이토치 JAVA < 텐서플로우 C
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
print(tf.__version__)
# %%
# 데이터 가져오기
# boston_housing머신러닝 딥러닝으로 어떻게 처리
# cifar10영상 사진 인식
# fashion_mnist그림 옷 인식
# reuters뉴스
mnist = tf.keras.datasets.mnist #(ex : 60000,28,28) => 28*28 = 784 레이어 늘어뜨리기
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
print(X_train.shape,X_test.shape)
print(Y_train.shape,Y_test.shape)

#%%
for i in range(10): 
    plt.imshow(X_train[i])
    plt.show()
    print(Y_train[i])

# %%
print('최대 : ',X_train[0].max(),'최소 :',X_train[0].min())
# %%
(x_train,y_train)=(X_train/255,Y_train)
(x_test,y_test)=(X_test/255,Y_test)
print(x_train[0].max())
plt.hist(y_train)
# %%
#모델결정 ANN
layers=[
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(10,activation='softmax') # n개 알고싶을 때, 2개는 시그모이드
]
model = tf.keras.models.Sequential(layers)
model.summary()
# %%
#최적화 함수 결정 : optimizer = 
#손실(에러)결정 : loss = 
model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %%
model.fit(x_train,y_train,epochs=10)
# %%
