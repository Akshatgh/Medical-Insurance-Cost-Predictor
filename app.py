import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open('medical_insurance_cost_predictor.sav', 'rb'))

# Prediction function
def medical_insurance_cost_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return round(prediction[0], 2)

def main():
    # Page configuration and styling
    st.set_page_config(page_title="Medical Insurance Cost Prediction", layout="centered")
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f2f6;
            color: #333333;
            font-family: Arial, sans-serif;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 5px;
        }
        .stButton button:hover {
            background-color: #45a049;
            color: #f0f2f6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Title and Introduction
    st.title('Medical Insurance Cost Prediction')
    st.markdown(
        "### Predict your annual medical insurance cost based on your profile.\n"
        "Fill out the details below, and click **Predict** to get an estimate."
    )

    # Input fields organized in two columns
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, step=1)
        bmi = st.number_input('Body Mass Index', min_value=10.0, max_value=60.0, step=0.1)
        children = st.number_input('Number of Children', min_value=0, max_value=10, step=1)

    with col2:
        sex = st.selectbox('Sex', ('Female', 'Male'))
        smoker = st.selectbox('Smoker', ('No', 'Yes'))
        region = st.selectbox('Region of Living', ('NorthEast', 'NorthWest', 'SouthEast', 'SouthWest'))

    # Mapping categorical inputs to numeric values
    sex = 1 if sex == 'Male' else 0
    smoker = 1 if smoker == 'Yes' else 0
    region = {'NorthEast': 0, 'NorthWest': 1, 'SouthEast': 2, 'SouthWest': 3}[region]

    # Predict Button and Output
    diagnosis = ''
    if st.button('Predict'):
        diagnosis = medical_insurance_cost_prediction([age, sex, bmi, children, smoker, region])
        st.success(f"Estimated Medical Insurance Cost: ${diagnosis}")

    # Footer message
    st.markdown(
        "<hr><center>Developed with ❤️ by Akshat</center>",
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
