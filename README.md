## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Abijith Shaan S
RegisterNumber: 212223080002 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")
dataset
dataset=dataset.drop("sl_no",axis=1)
dataset=dataset.drop("salary",axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes



dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset

X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values

Y

theta=np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
    
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
    
y_pred = predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

```

## Output:
Read the file and display
![image](https://github.com/Shaan2803/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/160568486/34084792-301a-43e3-8189-b76335888120)

Categorizing columns
![image](https://github.com/Shaan2803/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/160568486/d9cdd0ec-3714-47cd-a5f2-36751afd88f2)

Labelling columns and displaying dataset
![image](https://github.com/Shaan2803/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/160568486/c4de34a4-fe4c-4512-8f98-de4a8c31574d)

Display dependent variable
![image](https://github.com/Shaan2803/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/160568486/e6b100bb-e876-4b6f-9dec-6d4ead304b0b)

Printing accuracy
![image](https://github.com/Shaan2803/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/160568486/c3340c1e-3b1e-47d8-8b2f-fceb71bee841)

Printing Y
![image](https://github.com/Shaan2803/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/160568486/60f50131-98d8-4c19-97cb-b45827252d5f)

Printing y_prednew
![image](https://github.com/Shaan2803/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/160568486/47b6d871-21bd-4a66-8ec0-26cf97581785)












## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

