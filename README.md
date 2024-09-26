# Implementation-of-Linear-Regression-Using-Gradient-Descent

## Aim:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Step1:
Import the needed packages.
### Step2:
Read the txt file using read_csv.
### Step3:
Use numpy to find theta,x,y values.
### Step4:
To visualize the data use plt.plot.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: MONIKA RV
RegisterNumber: 212222220027
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("ex1.txt",header=None)

plt.scatter(df[0],df[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
    m=len(y) #length of the training data
    h=X.dot(theta) #hypothesis
    square_err=(h-y)**2
    
    return 1/(2*m)*np.sum(square_err) #returning J

df_n=df.values
m=df_n[:,0].size
X=np.append(np.ones((m,1)),df_n[:,0].reshape(m,1),axis=1)
y=df_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=[]

def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=[]
    for i in range(num_iters):
        predictions = X.dot(theta)
        error = np.dot(X.transpose(),(predictions -y))
        descent = alpha*(1/m )*error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
        return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

#Testing the implementation
theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(df[0],df[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0]for y in x_value]
plt.plot(x_value,y_value,color="purple")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions = np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000 , we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000 , we predict a profit of $"+str(round(predict2,0)))

```

## Output:
## Profit Prediction:
![o1](https://github.com/user-attachments/assets/b00d97ce-75e3-4eda-b974-777a7a234b98)


## Computecost:
![o2](https://github.com/user-attachments/assets/176953dd-a4bb-4df6-be62-c61c510d6252)

![o3](https://github.com/user-attachments/assets/45f187c8-bef1-4aa9-836c-2b6336531dfb)



## Cost Function using Gradient Descent:
![o4](https://github.com/user-attachments/assets/5c203714-2a88-4846-a462-50070fcd9805)

## Profit Prediction:
![o5](https://github.com/user-attachments/assets/9b5487cf-1b64-4876-87d9-ae71c54d03fc)

## 
![o6](https://github.com/user-attachments/assets/10a47c38-de17-411a-a4d5-2a71058236c2)

## 

![o7](https://github.com/user-attachments/assets/f36ab5f9-c345-4351-85c8-8c1f640f299f)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
