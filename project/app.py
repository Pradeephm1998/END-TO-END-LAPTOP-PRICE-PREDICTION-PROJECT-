# This is a sample Python script.
import numpy as np


import streamlit as st
import pickle
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error



import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

pipe = pickle.load(open('pipe1.pkl','rb'))
df = pickle.load(open('df1.pkl','rb'))


st.title("Laptop Predictor")

# brand
company = st.selectbox('Company Name',df['Company Name'].unique())

# type of laptop
type = st.selectbox('Type',df['Type'].unique())

# Ram
RAM = st.selectbox('RAM(in GB)',[4,8,16,32])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])


# screen size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
Processor_Name = st.selectbox('Processor Name',df['Processor_Name'].unique())




option = st.selectbox('Type of storge you lke to have',['EMMC','HDD AND SSD'])

if 'HDD AND SSD' in option:
        hdd = st.selectbox('HDD(in GB)',df['HDD_Capacity'].unique())
        ssd = st.selectbox('ssd(in GB)', df['SSD_Capacity'].unique())
else:
        EMMC = st.selectbox('EMMC(in GB)', df['EMMC'].unique())



gpu = st.selectbox('GPU',df['GPU'].unique())



MSOffice = st.selectbox('MSOffice',['No','Yes'])

os = st.selectbox('Operating_System',df['Operating_System'].unique())

if st.button('Predict Price'):
    # query

    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0



    if RAM == 32:
        RAM = 1
    elif RAM== 16:
        RAM = 2
    elif RAM == 8:
        RAM = 3
    elif RAM == 4:
        RAM = 4
    else:
        RAM=0
    if 'HDD AND SSD' in option:
        EMMC = 0
    else:
        hdd = 0
        ssd = 0
    if MSOffice == 'Yes':
        MSOffice = 1
    else:
        MSOffice = 0







    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([type,Processor_Name,RAM,os,touchscreen,weight,MSOffice,EMMC,ssd,hdd,company,gpu,ppi])
    query = query.reshape(1, 13)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))







