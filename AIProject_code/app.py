import streamlit as st
import pickle
from io import BytesIO
import requests
import pandas as pd 
# Code from Best Pipeline.py here


from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('https://raw.githubusercontent.com/dsuarezferre/AIproject/master/AIProject_code/prepared_data.csv')
features = tpot_data.drop('target', axis=1)
features1 = features.drop('fecha', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features1, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -398.43026381484333
exported_pipeline = XGBRegressor(learning_rate=0.1, max_depth=6, min_child_weight=1, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.45, verbosity=0)

exported_pipeline.fit(training_features, training_target)


######################
# User defined values
title = 'SO2 prediction for Zgz'
encoder_location = 'https://github.com/dsuarezferre/AIproject/blob/master/AIProject_code/encoder.pkl?raw=true'
target_encoder_location = ''
if len(encoder_location) > 5:
    mfile = BytesIO(requests.get(encoder_location).content)
    encoder = pickle.load(mfile)
    df = encoder.inverse_transform(features)
else:
    df = features.copy()
if len(target_encoder_location) > 5:
    mfile = BytesIO(requests.get(target_encoder_location).content)
    target_encoder = pickle.load(mfile)
st.title(title)
st.sidebar.header('User Input Parameters')
st.subheader('User Input parameters')
selected_data = dict()
for column in df.columns[1:]:
    if column != 'target' or column != 'fecha':
        label = column.replace('_id.','')
        label = label.replace('_',' ').title()
        if df[column].dtype == 'O':
            selected_value = st.sidebar.selectbox(label, list(df[column].unique()))
        elif df[column].dtype == 'int64':
            selected_value = st.sidebar.number_input(label, min_value=df[column].min(), max_value=df[column].max(), value=df[column].iloc[0], step=1)
        elif df[column].dtype == 'float64':
            selected_value = st.sidebar.number_input(label, min_value=df[column].min(), max_value=df[column].max(), value=df[column].iloc[0])
        
        selected_data[column] = selected_value
test_data = pd.DataFrame(selected_data, index=[0])
test_data['fecha']= '2017-01-01T00:00:00+00:00'
st.write(test_data)
st.write(features.head(1))
st.subheader('Prediction')
if len(encoder_location) > 5:
    test_data = encoder.transform(test_data) 
prediction = exported_pipeline.predict(test_data)
if len(target_encoder_location) > 5:
    prediction = target_encoder.inverse_transform(prediction)
if 'float' in str(type(prediction[0])):
    st.write(round(prediction[0],2))
else:
    st.write(prediction[0])
    
