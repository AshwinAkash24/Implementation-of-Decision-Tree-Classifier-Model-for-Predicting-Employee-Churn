# EX-08:Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### Step-1:
Read the dataset from a CSV file.
#### Step-2:
Convert categorical 'salary' column to numeric using Label Encoding.
#### Step-3:
Select input features (x) and target variable (y).
#### Step-4:
Split the data into training and testing sets.
#### Step-5:
Train a Decision Tree Classifier using the training data.
#### Step-6:
Predict outcomes and calculate accuracy of the model.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Ashwin Akash M
RegisterNumber:  212223230024
*/
import pandas as pd
df=pd.read_csv("Employee.csv")
df
df.info()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["salary"]=le.fit_transform(df['salary'])
df.head()
x=df[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
y=df[['left']]
print(x)
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
ypred=dt.predict(x_test)
print(ypred)
print(y_test)
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,ypred)
print(acc)
dt.predict([[.5,.8,9,260,6,0,1,3]])
```

## Output:
![image](https://github.com/user-attachments/assets/79a79989-8392-414c-8e0e-5c6eac118457)<br>
![image](https://github.com/user-attachments/assets/da861015-a250-47f8-b608-71746065ffa7)<br>
![image](https://github.com/user-attachments/assets/9181863a-27fb-4d9f-89dd-56a83d69c186)<br>
![image](https://github.com/user-attachments/assets/ad96c10d-7ff0-473d-a376-bf4cb5356241)<br>
![image](https://github.com/user-attachments/assets/27bb6a7b-b775-476c-a443-f4bbb0a0f21c)<br>
![image](https://github.com/user-attachments/assets/11dfb6f7-0edc-4199-a1df-4555a35ff49c)<br>
![image](https://github.com/user-attachments/assets/33e40d65-50c4-4689-992b-870ff8c1c949)<br>
![image](https://github.com/user-attachments/assets/f6f9c040-648f-468d-b0e1-9ac4eb45e089)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
