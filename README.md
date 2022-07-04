# Diabetes-Prediction-Model-
Diabetes Prediction Model  using SVM Model

# Diabetes Prediction Using SVM Model

## importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("diabetes.csv")
df.head()

df.shape

df.describe()

# Testing for the correlataion between the dataset features. 
sns.heatmap(data=df.corr(),cmap="YlGnBu", annot=True)
plt.show()

## EDA PERFORMANCE

df.isna().sum()

df["Outcome"].value_counts()

df.groupby("Outcome").mean()

### Splitting the columns into X and Y Variables for the model building process.

X = df.drop("Outcome", axis=1)
y= df["Outcome"]

X.head()

y.head()

# As the scale of all the column values are not in same scale. So we need to Standardize the values to one scale.
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

#fitting the attribute dataset to the scaling library.
sc.fit(X)

#transforming the datsets into scaled arrays  
X = sc.transform(X)

X

y

### Splitting the dataset into Training and Testing Dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

X_train

y_train

X_test

y_test

X_train.shape

X_test.shape

y_train.shape

y_test.shape

## Model Building 

from sklearn import svm

#attaining a variable to the model 
classifier = svm.SVC(kernel='linear')

#fitting the training dataset into model
classifier.fit(X_train, y_train)

#predicting the output for the training dataset
X_train_pred = classifier.predict(X_train)

from sklearn.metrics import accuracy_score

#Calculating how accurate the model is built according to the actual datset 
X_train_accuracy = accuracy_score(X_train_pred, y_train)

print("The Training model Prediction is :", round(X_train_accuracy *100,2), "%")

#Now modelling the same with the Test Datastet
X_test_pred = classifier.predict(X_test)

#Calculating how accurate the model is built according to the actual datset
X_test_accuracy = accuracy_score(X_test_pred, y_test)

print("The Testing model Prediction is :", round(X_test_accuracy *100,2), "%")

## Manual Prediction 

input_data = (2,197,70,45,543,30.5,0.158,53)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

std_data = sc.transform(input_data_reshaped)

std_data

prediction = classifier.predict(std_data)

print(prediction)

if (prediction ==1):
    print("The person is a Diabetic")
else:
    print("the person is not a Diabetic")


## Thus the SVM Model for detecting any Diabetic Patient is Ready and accurate. 
