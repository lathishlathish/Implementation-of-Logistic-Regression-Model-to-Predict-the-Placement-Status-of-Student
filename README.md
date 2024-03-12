# EXP:4 Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: M.LATHISH KANNA
RegisterNumber: 212222230073
*/
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull()

data1.duplicated().sum()

from sklearn .preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![image](https://github.com/lathishlathish/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120359170/a8e6f266-2eaf-49eb-b5a7-60d86f78aef3)
![image](https://github.com/lathishlathish/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120359170/c9127f12-67c4-4718-991d-2366546f09f4)
![image](https://github.com/lathishlathish/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120359170/f68e6b65-2755-4af4-99bf-80b88ce8a2fb)
![image](https://github.com/lathishlathish/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120359170/8080ba86-588a-48bd-8570-da961e1f47fd)
![image](https://github.com/lathishlathish/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120359170/de1a0ea5-839e-40fd-88b6-e2000ca4877c)
![image](https://github.com/lathishlathish/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120359170/8b902baa-d208-428e-99e2-9e8ca3f97444)
![image](https://github.com/lathishlathish/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120359170/245350d2-d6d9-48b1-9571-c1a22c7e007f)











## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
