import numpy as np
import pandas as pd
import pickle

#Renaming DiabetesPedigreeFunction as DPF
df=pd.read_csv('diabetes.csv')
df=df.rename(columns={'DiabetedPedigreenFunction':'DPF'})

#Replacing the 0 values['Glucose,'BloodPressure','SkinThickness','Insulin','BMI']
df_copy=df.copy(deep=True)
df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replce(0,np.NaN)

#Replacing NaN value by mean,median depending upon distribution
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(),inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(),inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].mean(),inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].mean(),inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].mean(),inplace=True)

#Model Building
from sklearn.model_selection import train_test_split
X=df.drop(columns='Outcome')
y=df['Outcome']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#Creating Random Forest Model
