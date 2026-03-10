import streamlit as st
import pickle
import pandas as pd
from pathlib import Path
# Resolve all files relative to app.py (important on Streamlit Cloud)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'insurance_model.pkl'
DATA_PATH = BASE_DIR / 'insurance.csv'
# Load model
try:    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except ModuleNotFoundError as e:
    st.error('A required Python package is missing in deployment. Add it to requirements.txt and redeploy.')
    st.exception(e)
    st.stop()
except FileNotFoundError as e:
    st.error('Model artifact file is missing. Ensure .pkl files are committed and deployed with app.py.')
    st.exception(e)
    st.stop()
st.title('Insurance Charges Prediction')
df = pd.read_csv(DATA_PATH)
st.write(df.head())
st.subheader('Enter Insurance Details')
age = st.number_input('Age', min_value=0, max_value=120)
sex = st.selectbox('Sex', ['male', 'female'])
bmi = st.number_input('BMI', min_value=0.0, max_value=100.0)
children = st.number_input('Number of Children', min_value=0, max_value=10)
smoker = st.selectbox('Smoker', ['yes', 'no'])
region = st.selectbox('Region', df['region'].unique())
if st.button('Predict Charges'):
    sex_map = {'male': 0, 'female': 1}
    smoker_map = {'yes': 0, 'no': 1}
    region_map = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    input_data['sex'] = input_data['sex'].map(sex_map)
    input_data['smoker'] = input_data['smoker'].map(smoker_map)
    input_data['region'] = input_data['region'].map(region_map)
    try:
        SCALER_PATH = BASE_DIR / 'scaler.pkl'
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        input_data = scaler.transform(input_data)
    except FileNotFoundError:
        pass
    prediction = model.predict(input_data)
    st.success(f'Predicted Insurance Charges:${prediction[0]:.2f}')