import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
from logistic_regression import LogisticRegression
import numpy as np

# loading data from a file 
diabetes_data=pd.read_csv("data/diabetes.csv")

############## printing out the 5 first rows 
# print(diabetes_data.head())


############## printing out the size of dataframe 
# print(diabetes_data.shape)

############## how much null values in this dataset
# print(diabetes_data.isnull().sum())

############## calculating diffrents values in Outcome 
# print(diabetes_data["Outcome"].value_counts())

############## caclulating the average of each feature for each categories diabetic or non-diabetic
print(diabetes_data.groupby("Outcome").mean())

############## separating target and feature 
features = diabetes_data.drop(columns=["Outcome"],axis=1).values
target = diabetes_data["Outcome"].values
# print("Features before standardization \n",features)

############## creating instance of StandardScaler
standard_scaler=StandardScaler()
standard_scaler.fit(X=features)
features=standard_scaler.transform(X=features)
# print("\n\nFeatures after standardization \n",features)

############## splitting dataset to training and test data
X_train,X_test,Y_train,Y_test=train_test_split(features,target,test_size=0.2,random_state=2)


############## creating instance of our model
classifier=LogisticRegression(learning_rate=0.01,no_of_iteration=1000)

############## training step
classifier.fit(X_train,Y_train)

############## trying to predict 
Y_predicted=classifier.predict(X_test)


# print("\nthe outcome of test :\n",Y_test)
# print("\nthe predicted outcome :\n",Y_predicted)

############## compare Y_predicted to Y_test
score=accuracy_score(Y_test,Y_predicted,normalize=True)
print("the accuracy score of the logistic regression model is : ",score*100)


############## making a predictive system 
# input_data=(5,166,72,19,175,25.8,0.587,51)
input_data=(2,197,70,45,543,30.5,0.158,53)

# changing the forma to numpy array
input_data_nmp_array=np.asanyarray(input_data)

# transforming input to one row 
input_data_reshaped=input_data_nmp_array.reshape(1,-1)

# standardization of data 
std_input=standard_scaler.transform(input_data_reshaped)

# making prediction 
prediction=classifier.predict(input_data_reshaped)

if prediction[0]==0:
    print("the passion is not diabetic")
else:
    print("the passion is diabetic")




# displaying data 

X_axis = np.linspace(features[:, 1].min(), features[:, 1].max(), 300).reshape(-1, 1)
z = X_axis * classifier.w[1] + classifier.b
probs = 1 / (1 + np.exp(-z))

plt.figure(figsize=(10, 6))
plt.scatter(features[:, 1], target, color='red', alpha=0.3, label='Actual Data')
plt.plot(X_axis, probs, color='blue', linewidth=3, label='Logistic Curve')
plt.title("Probability of Diabetes vs Glucose")
plt.xlabel("Standardized Glucose")
plt.ylabel("Probability")
plt.legend()
plt.show()