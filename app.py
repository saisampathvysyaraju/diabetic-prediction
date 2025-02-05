import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


diabetes_dataset = pd.read_csv('diabetes.csv')


X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)


def calculate_bmi(weight, height_feet):
    height_meters = height_feet * 0.3048  
    return round(weight / (height_meters ** 2), 2)

st.title("Diabetes Prediction App")
st.write("### Enter the required details to check for diabetes prediction")


st.write("""
#### Instructions:
- **Pregnancies**: Number of times pregnant (0 if not applicable).
- **Glucose Level**: Blood glucose concentration (0-300 mg/dL).
- **Blood Pressure**: Blood pressure measurement (0-200 mm Hg).
- **Skin Thickness**: Thickness of skin fold at the triceps (0-100 mm).
- **Insulin Level**: Blood insulin level (0-900 IU/mL).
Diabetes Pedigree Function (DPF)** represents genetic risk based on family history.
- **Enter the value based on your family history:**
  - **0.0**: No family history of diabetes.
  - **0.1 - 1.0**: Mild to moderate family history (e.g., one parent or sibling with diabetes).
  - **Above 1.0**: Strong family history (e.g., both parents or multiple relatives with diabetes)


BMI will be automatically calculated based on weight and height.
""")


gender = st.selectbox("Gender", ["Male", "Female"])


pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1, value=0, disabled=(gender == "Male"))
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, step=1, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, step=1, value=80)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, step=1, value=30)
weight = st.number_input("Weight (kg)", min_value=1.0, max_value=200.0, step=0.1, value=70.0)
height_feet = st.number_input("Height (feet)", min_value=1.0, max_value=8.0, step=0.1, value=5.8)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, step=1, value=30)


bmi = calculate_bmi(weight, height_feet)
st.write(f"**Calculated BMI:** {bmi}")


if st.button("Predict"):  
    input_data = np.array([[pregnancies if gender == "Female" else 0, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = classifier.predict(input_scaled)
    
    if prediction[0] == 0:
        st.success("The person is not likely diabetic.")
        st.write('Suggested Actions:')
        st.write('-🥗 Maintain a healthy diet rich in vegetables, fruits, and whole grains.')
        st.write('-🏃‍♂️ 🚴 🏋️‍♀️Keep active with regular exercise.')
        st.write('-📊 📉 Monitor your weight and aim for a healthy BMI.')
       
    else:
        st.error("The person is likely diabetic.")
        st.write('Suggested Actions:')
        st.write('- 🏥Consult a healthcare provider for proper diagnosis and management.')
        st.write('- 🥗Follow a balanced diet with low sugar and processed foods.')
        st.write('- 🏃‍♂️ 🚴 🏋️‍♀️Engage in regular physical activity, especially aerobic exercise.')
        st.write('- 📊 📉Monitor blood sugar levels regularly.')
    
 
    st.info("**Note:** This prediction is based on a machine learning model trained on high-quality data. However, it may not always be accurate. Consult a medical professional for a proper diagnosis.")
