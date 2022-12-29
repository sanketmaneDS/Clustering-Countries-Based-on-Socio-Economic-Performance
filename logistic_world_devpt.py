import pandas as pd 
import streamlit as st 
from sklearn.linear_model import LogisticRegression 
from pickle import dump
from pickle import load

st.title('Model Deployment: World Development')

st.sidebar.header('User Input Parameters')

def user_input_features():
    Birth_Rate = st.sidebar.number_input("Insert Birth Rate")
    CO2_Emissions= st.sidebar.number_input("Insert CO2 emissions")
    Days_to_Start_Business = st.sidebar.number_input("Insert Days to Start Business")
    Energy_Usage = st.sidebar.number_input("Insert Energy Usage")
    GDP_in_USD = st.sidebar.number_input("Insert GDP_$")
    Health_Exp_in_pc_GDP = st.sidebar.number_input("Health Exp % GDP")
    Health_Exp_per_Capita_in_USD = st.sidebar.number_input("Insert Health Exp/Capita_$")
    IMR = st.sidebar.number_input("Insert Infant Mortality Rate")
    Internet_Usage = st.sidebar.number_input("InsertInternet Usage")
    Lending_Interest= st.sidebar.number_input("Insert Lending Interest")
    Life_Expectancy_Female = st.sidebar.number_input("Insert Life Expectancy Female")
    Life_Expectancy_Male = st.sidebar.number_input("Life Expectancy Male")
    Mobile_Phone_Usage = st.sidebar.number_input("Insert Mobile Phone Usage")
    Population_0_14 = st.sidebar.number_input("Population 0-14")
    Population_15_64 = st.sidebar.number_input("Insert Population 15-64")
    Population_65 = st.sidebar.number_input("Population 65")
    Population_Total = st.sidebar.number_input("Insert Population Total")
    Population_Urban= st.sidebar.number_input("Insert Population Urban")
    Tourism_Inbound_USD = st.sidebar.number_input("Insert Tourism Inbound_$")
    Tourism_Outbound_USD = st.sidebar.number_input("Insert Tourism Outbound_$")
    
    
    data = {'Birth Rate':Birth_Rate,
            'CO2 Emissions':CO2_Emissions,
            'Days to Start Business':Days_to_Start_Business,
            'Energy Usage':Energy_Usage,
            'GDP_$':GDP_in_USD,
            'Health Exp % GDP': Health_Exp_in_pc_GDP,
            'Health Exp/Capita_$':Health_Exp_per_Capita_in_USD,
            'Infant Mortality Rate':IMR,
            'Internet Usage':Internet_Usage,
            'Lending Interest':Lending_Interest,
            'Life Expectancy Female':Life_Expectancy_Female,
            'Life Expectancy Male':Life_Expectancy_Male,
            'Mobile Phone Usage':Mobile_Phone_Usage,
            'Population 0-14':Population_0_14,
            'Population 15-64':Population_15_64,
            'Population 65+':Population_65,
            'Population Total':Population_Total,
            'Population Urban':Population_Urban,
            'Tourism Inbound_$':Tourism_Inbound_USD,
            'Tourism Outbound_$':Tourism_Outbound_USD,
           }
    features = pd.DataFrame(data,index = [0])
    return features 
  
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


# load the model from disk
loaded_model = load(open('log.pkl', 'rb'))

prediction = loaded_model.predict(df)

st.subheader('Predicted Result')

st.write('ClusterID', prediction)

if prediction==0:
    string = 'Asian Rising Stars'
elif prediction==1:
    string ='small economies but developed'
elif prediction==2:
    string ='Under developed'
elif prediction==3:
    string ='World Hegemon'
elif prediction==4:
    string ='European and asian developed countries'

st.write('Country Development:', string)

st.write(pd.DataFrame({
    'CLusterID': [0, 1, 2, 3, 4],
    'Development': ['Asian Rising Stars','small economies but developed',
                   'Under developed','World Hegemon', 'European and asian developed countries'],
}))


