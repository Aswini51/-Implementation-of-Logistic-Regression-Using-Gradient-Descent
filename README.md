# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary.
6. Define a function to predict the Regression value.


## Program and Output:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: G ASWINI
RegisterNumber:  212224040037
*/

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

data=pd.read_csv("/content/Placement_Data.csv")
data.head()
```
![Screenshot 2025-04-14 144202](https://github.com/user-attachments/assets/e576634b-aec7-49cb-ab51-f6ad9c55f458)

```
data = data.drop(['sl_no', 'salary'], axis=1)
data
```

![Screenshot 2025-04-14 144220](https://github.com/user-attachments/assets/6e0dd067-8d46-414d-a395-06b3962c73e6)

```
data["gender"]=data["gender"].astype('category') 
data["ssc_b"]=data["ssc_b"].astype('category') 
data["hsc_b"]=data["hsc_b"].astype('category') 
data["degree_t"]=data["degree_t"].astype('category') 
data["workex"]=data["workex"].astype('category') 
data["specialisation"]=data["specialisation"].astype('category') 
data["status"]=data["status"].astype('category') 
data["hsc_s"]=data["hsc_s"].astype('category') 
data.dtypes
```

![Screenshot 2025-04-14 144248](https://github.com/user-attachments/assets/589cf1eb-86d2-405d-914b-1eba906af209)

```
data["gender"]=data["gender"].cat.codes 
data["ssc_b"]=data["ssc_b"].cat.codes 
data["hsc_b"]=data["hsc_b"].cat. codes
data["degree_t"]=data["degree_t"].cat.codes 
data["workex"]=data["workex"].cat.codes 
data["specialisation"]=data["specialisation"].cat.codes 
data["status"]=data["status"].cat.codes 
data["hsc_s"]=data["hsc_s"].cat.codes 
data
```

![Screenshot 2025-04-14 144311](https://github.com/user-attachments/assets/1cc7f5bf-b569-4b64-a9a0-086d504809c0)

```
x = data.iloc[:, :-1].values 
y = data.iloc[:, -1].values 
y
```

![Screenshot 2025-04-14 145429](https://github.com/user-attachments/assets/a06a6b6c-b57e-43eb-ad58-4d6aa39fe89f)

```
theta = np.random.randn(x.shape[1]) 
Y = y
def sigmoid(z): 
    return 1 / (1 + np.exp(-z))
def loss(theta, X, y): 
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
def gradient_descent(theta, X, y, alpha, num_iterations): 
    m = len(y)
    for i in range(num_iterations): 
        h = sigmoid(X.dot(theta)) 
        gradient = X.T.dot(h - y) / m 
        theta -= alpha * gradient 
    return theta
theta = gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)
def predict(theta, X): 
    h = sigmoid(X.dot(theta)) 
    y_pred=np.where(h>=0.5,1,0) 
    return y_pred

y_pred = predict(theta, x) 
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy: ", accuracy) 
print(y_pred)
```

![Screenshot 2025-04-14 145438](https://github.com/user-attachments/assets/51bad520-61a1-424c-9916-cef3bc6af735)

```
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]]) 
y_prednew = predict(theta, xnew) 
print(y_prednew)
```

![Screenshot 2025-04-14 144542](https://github.com/user-attachments/assets/40909e1a-dcd8-4273-aa21-d34f978bc02f)

```
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]]) 
y_prednew = predict(theta, xnew) 
print(y_prednew)
```

![Screenshot 2025-04-14 144548](https://github.com/user-attachments/assets/d47ec8c9-d605-43cd-9dc9-511c1c2c8c67)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

