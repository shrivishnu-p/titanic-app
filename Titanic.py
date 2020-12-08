import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from joblib import dump, load

st.set_page_config(page_title='Titanic Survivor Prediction', page_icon='🚢', layout='wide')
st.header('Titanic Survivor Prediction')

model = load('titanicSVM.pkl')

# Form Layout
a1, a2, a3 = st.beta_columns(3)
b1, b2, b3 = st.beta_columns(3)
c1, c2 = st.beta_columns(2)

with a1:
  Sex0 = st.selectbox('Select Gender of the Passenger', ['Male', 'Female'])
with a2:
  Age0 = st.number_input('Enter the Age of the Passenger', min_value=0, max_value=100, value=20)
with a3:
  Pclass = st.selectbox('Select Passenger Class', [1,2,3])
with b1:
  Fare0 = st.number_input('Enter the Fare paid by the Passenger', min_value=0, max_value=150, value=10)
with b2:
  Cabin0 = st.selectbox('Select the Cabin', ['A','B','C','D','E','F','G','T'])
with b3:
  Embarked0 = st.selectbox('Select the Location of Embarkment', ['Southampton','Cherbourg','Queenstown'])
with c1:
  FamilySize0 = st.number_input('Select the number of Persons from his/her Family Travelled in Titanic', 1, 11)
if Sex0 == 'Male':
  Title_x = ['Mr','Master','Dr','Rev','Col','Major','Jonkheer','Don','Capt','Sir']
else:
  Title_x = ['Miss','Mrs','Dr','Mlle','Countess','Ms','Lady','Dona','Mme']
with c2:
  Title0 = st.selectbox('Select the Title Used (if any)', Title_x)

# Mapping Sex
sex_mapping = {'Male': 0, 'Female': 1}
Sex = sex_mapping.get(Sex0)

# Mapping Age
Age = Age0
if Age0 in range(0,17):
    Age = 0
elif Age0 in range(17,27):
    Age = 1
elif Age0 in range(27,37):
    Age = 2
elif Age0 in range(37,63):
    Age = 3
else:
    Age = 4

# Mapping Fare
Fare = Fare0
if Fare in range(0,18):
    Fare = 0
elif Fare in range(18,31):
    Fare = 1
elif Fare in range(31,101):
    Fare = 2
else:
    Fare = 3

# Mapping Cabin
cabin_mapping = {'A': 0, 'B': 0.4, 'C': 0.8, 'D': 1.2, 'E': 1.6, 'F': 2, 'G': 2.4, 'T': 2.8}
Cabin = cabin_mapping.get(Cabin0)

# Mapping Embarked
embarked_mapping = {'Southampton': 0, 'Cherbourg': 1, 'Queenstown': 2}
Embarked = embarked_mapping.get(Embarked0)

# Family Size
FamilySize = FamilySize0 + 1

# Title Mapping
title_mapping = {'Mr': 0, 'Miss': 1, 'Mrs': 2,
                 'Master': 3, 'Dr': 3, 'Rev': 3, 'Col': 3, 'Major': 3, 'Mlle': 3,'Countess': 3,
                 'Ms': 3, 'Lady': 3, 'Jonkheer': 3, 'Don': 3, 'Dona' : 3, 'Mme': 3,'Capt': 3,'Sir': 3}
Title = title_mapping.get(Title0)

# Creating Dataframe
test_dict = {'Pclass' : Pclass,
             'Sex' : Sex,
             'Age' : Age,
             'Fare' : Fare,
             'Cabin' : Cabin,
             'Embarked' : Embarked,
             'Title' : Title,
             'FamilySize' : FamilySize}
test_data = pd.DataFrame({0:test_dict}).T
# st.write(test_data)

# Predicting the survival
pred = model.predict(test_data)
# st.write(pred)

# Displaying the prediction
if st.button('Predict'):
    if pred[0] == 1:
        st.write(f'The {Sex0} passenger of age {Age0}, who embarked at {Embarked0} after purchasing a class '
                 f'{Pclass} ticket of cost ${Fare0} accompanied by {FamilySize0} of his family members has survived.')
        st.success('The passenger survived the epic Titanic Voyage! 😃')
        st.balloons()

    elif pred[0] == 0:
        st.write(f'The {Sex0} passenger of age {Age0}, who embarked at {Embarked0} after purchasing a class '
                 f'{Pclass} ticket of cost ${Fare0} accompanied by {FamilySize0} of his family members didnot survive.')
        st.error('Sorry, the passenger did not survive 😢')


if st.button('Reset'):
    pred = None
