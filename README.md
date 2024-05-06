# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start.
2. Import the standard Libraries. 
3. Set variables for assigning dataset values. 
4. Import linear regression from sklearn. 
5. Assign the points for representing in the graph. 
6. Predict the regression for marks by using the representation of the graph.
7. Stop.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ROHITH PREM S
RegisterNumber:  212223040172
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/SEC/Downloads/student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
```
```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours Vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_train,regressor.predict(X_train),color='yellow')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)  
```

## Output:

### df.head()
![head](https://github.com/23003250/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331462/3c5080ef-6535-4d88-b010-c67bd2998b72)
<br><br><br><br>





### df.tail()
![tail](https://github.com/23003250/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331462/e9decb0a-b79b-4441-9faf-f15443e3541d)

### Array value of X
![array_value](https://github.com/23003250/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331462/918413cc-f7c6-486c-a72e-2f3d4c83c179)
<br><br><br><br>





### Array value of Y
![array y](https://github.com/23003250/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331462/9ee52757-9cc1-4093-b01a-925d69375611)

### Values of y prediction
![y prediction](https://github.com/23003250/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331462/3ac730f0-6054-4fc7-92dc-7a7c4b3a52b8)

### Array values of Y test
![array value of y test](https://github.com/23003250/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331462/4152c892-4edc-4c2f-ba6a-a0450563fbd2)

### Training set graph
![training set graph](https://github.com/23003250/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331462/7af539a4-ffe7-4ed8-897c-61cff9425abb)
<br><br><br><br><br><br><br>








### Test set graph
![test set graph](https://github.com/23003250/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331462/9e34602b-4b76-4422-bcee-850c2d055d87)

### Values of MSE,MAE and RMSE
![values of mse,mae and rmse](https://github.com/23003250/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331462/b2d203af-4572-4f37-a6c9-0fe8dd0ca86c)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
