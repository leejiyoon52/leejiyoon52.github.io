---
layout: post
title: Support Vector Regression
tags: [data analytics]
modified: 2018-11-17
use_math: true
image:
  path: /images/abstract-7.jpg
  feature: abstract-7.jpg

---

Kernel-based Learning: Support Vector Regression
=======

---
### 비선형 데이터를 활용한 예시
---

#### SVR의 각 요소를 비교 및 확인
SVR의 경우 고려해야하는 Loss function과 Kernel function, 하이퍼파라미터가 다양하게 존재합니다. 따라서 이들을 변화시키며 결과를 확인해보도록 하겠습니다.


#### **1. 랜덤 데이터 생성**

 삼각함수를 사용하여 비선형성을 갖는 데이터를 생성하고, 일부 난수에 대해 노이즈를 추가해봅시다.

```python
X = np.sort(10*np.random.rand(100,1), axis=0) # X : 0-10사이 난수를 100개 생성
y = np.sin(X).ravel()                         # y : sin(X)를 통해 비선형 데이터 생성

y[::5] +=2*(0.5-np.random.rand(20))           # 일부 y값에 NOISE 추가
y[::4] +=3*(0.5-np.random.rand(25))
y[::1] +=1*(0.5-np.random.rand(100))
```

생성된 데이터의 개형은 다음과 같습니다. 데이터들이 sin(X)함수를 기준으로 노이즈가 반영되어 적절히 흩어지게 되었습니다. 해당 데이터를 잘 반영하여 새로운 데이터를 잘 예측하는 회귀선을 구하고자 하는 것이 SVR의 목적이라 할 수 있습니다.
<p align="center"><img width="500" height="auto" img src="/images/image_1.png"></p>


#### **2. Kernel function 비교**

#####Kernel function

앞서 소개했듯이 대표적인 커널함수(kernel function)는 **(1)Linear kernel (2)Polynomial kernel (3)RBF kernel** 이 있으며, 이들을 구현한 코드는 다음과 같습니다. 코드 상에서 함수의 하이퍼 파라미터 'coef0'는 linear, polynomial, sigmoid kernel에서의 bias값을 의미하며, 'gamma'는 RBF, sigmoid kernel에서 $1/\sigma^2$을 의미합니다. 'gamma'로 치환하므로써 연산을 보다 용이하게 개선할 수 있습니다.

```python
def kernel_f(xi, xj, kernel = None, coef0=1.0, degree=3, gamma=0.1):

    if kernel == 'linear':                                  # Linear kernel
        result = np.dot(xi,xj)+coef0
    elif kernel == 'poly':                                  # Polynomial kernel
        result = (np.dot(xi,xj)+coef0)**degree
    elif kernel == 'rbf':                                   # RBF kernel
        result = np.exp(-gamma*np.linalg.norm(xi-xj)**2)
    elif kernel =='sigmoid':                                # Sigmoid kernel
        result = np.tanh(gamma*np.dot(xi,xj)+coef0)
    else:                                                   # Dot product
        result = np.dot(xi,xj)

    return result
```
<br />
##### Kernel matrix
두 원소값에 대한 스칼라 $K(x_i,x_j)$를 구할 수 있으며, 모든 원소간(Pair-wise) $K$값을 구해 Kernel matrix(Gram matrix)를 도출해야합니다. 최종적으로 도출되는 커널행렬(Kernel matrix)은 **대칭행렬(Symmetric matrix)** 이고, 모든 $K$값이 양수인 **Positive semi-definite 행렬** 이라는 특징을 갖고있습니다. 아래 코드를 통해 앞서 정의한 커널함수에 따른 커널행렬을 구합니다.  


```python
def kernel_matrix(X, kernel, coef0=1.0, degree=3, gamma=0.1):

    X = np.array(X,dtype=np.float64)
    mat = []
    for i in X:
        row = []
        for j in X:
            if kernel=='linear':           
                row.append(kernel_f(i, j, kernel = 'linear', coef0))   
            elif kernel=='poly':
                row.append(kernel_f(i, j, kernel = 'poly', coef0, degree))
            elif kernel=='rbf':
                row.append(kernel_f(i,j,kernel = 'rbf', gamma))
            elif kernel =='sigmoid':
                row.append(kernel_f(i,j,kernel = 'sigmoid', coef0, gamma))    
            else:
                row.append(np.dot(i,j))
        mat.append(row)

    return mat
```

커널함수만을 변형시켜 회귀계수 추정을 비교해봅시다. 커널함수비교를 위해 손실함수는 epsilon insensitive로, 하이퍼 파라미터는 (epsilon=1, C=0.001, gamma=0.1)로 정의했습니다. 한눈에 보기에도 해당 데이터에는 RBF kernel을 사용한 회귀분석이 가장 적합한 것을 확인할 수 있었습니다. 이처럼 주어진 데이터와 문제상황에 따라 최적의 커널함수를 직접 학습하고, 실험하여 찾아야합니다.


<p align="center"><img width="650" height="auto" img src="/images/image_2.png"></p>

#### **3. Loss function 비교**

##### Loss function
흔히 선형회귀문제에서는 손실함수(Loss function)를 MSE(Mean squared error)로 정의하여 사용합니다. 하지만

<br />
##### Loss function hyperparameter

기존의 선형회귀와 가장 큰 관점차이는 손실함수(Loss function)에 Penalty(C)를 부여한다는 점입니다.


$$
\min { \frac { 1 }{ 2 } { \left\| w \right\|  }^{ 2} } +C\sum _{ i=1 }^ n {({ \xi  }_{ i }+{\xi}_{i}^{*})}
$$

$$
\\s.t. \quad    ({ w }^{ T }{ x }_{ i }+b)-{ y }_{ i }\le {\epsilon}+{ \xi  }_{ i }
$$

$$
y_i-(w^Tx_i +b) \le {\epsilon}+{ \xi }_i^*
$$

$$
\\ { \xi  }_{ i }, { \xi  }_{ i }^* \ge {0}
$$



##Primal Lagrangian



$$
{L_{p}} =  { \frac { 1 }{ 2 } { \left\| w \right\|  }^{ 2} } + C\sum _{ i=1 }^{ n }{ ({ \xi  }_{ i }+{\xi}_{i}^{*})} - \sum _{ i=1 }^{ n }{ ({\eta}_{i}{ \xi  }_{ i }+{\eta}_{i}^{*}{\xi}_i^* )}
$$

$$
\\-\sum _{ i=1 }^{ n }{ {\alpha}_{i}({ \epsilon }+{\xi}_{i}+{y}_{i}-{W}^{T}{x}_{i}-b)} - \sum _{ i=1 }^{ n }{ {\alpha}_{i}^{*}({ \epsilon }+{\xi}_{i}^* -{y}_{i}-{W}^{T}{x}_{i}+b)}
$$

$$
 {\alpha}_{i}^* ,{\eta}_{i}^* \ge 0
$$



#### **3. Loss function 비교**

###Primal Lagrangian
