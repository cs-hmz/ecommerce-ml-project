import streamlit as st
import joblib
import numpy as np

# Charger le modèle
model = joblib.load("models/model.pkl")

st.title("Ecommerce Customer Spending Prediction")

st.write("Entrez les caractéristiques du client pour prédire ses dépenses annuelles.")

# Champs d'entrée
avg_session_length = st.number_input("Average Session Length", min_value=0.0, step=0.1)
time_on_app = st.number_input("Time on App", min_value=0.0, step=0.1)
time_on_website = st.number_input("Time on Website", min_value=0.0, step=0.1)
length_of_membership = st.number_input("Length of Membership", min_value=0.0, step=0.1)

if st.button("Prédire"):
    features = np.array([[avg_session_length, time_on_app, time_on_website, length_of_membership]])
    prediction = model.predict(features)
    st.success(f"Prédiction des dépenses annuelles : {prediction[0]:.2f} $")
