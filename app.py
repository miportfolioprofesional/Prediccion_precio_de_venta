import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Cargar el modelo entrenado
model = joblib.load("modelo_lineal_regresion_car_prices.pkl")

# Cargar el target encoder para las variables categóricas
target_encoder = joblib.load("target_encoder.pkl")

# Título de la aplicación
st.title("Predicción del precio de vehículos")

# Entradas para las características
year = st.number_input('Año', min_value=1990, max_value=2024, value=2015)
make = st.text_input('Marca (Ej: Kia, BMW)')
model_name = st.text_input('Modelo (Ej: Sorento, 3 Series)')
trim = st.text_input('Versión (Ej: LX, 328i SULEV)')
body = st.text_input('Tipo de carrocería (Ej: SUV, Sedan)')
state = st.text_input('Estado (Ej: ca, wa)')
condition = st.number_input('Condición (1.0 - 5.0)', min_value=1.0, max_value=5.0, value=3.0)
odometer = st.number_input('Kilometraje', min_value=0.0, value=15000.0)
color = st.text_input('Color exterior (Ej: white, black)')
interior = st.text_input('Color interior (Ej: black, beige)')
mmr = st.number_input('Valor MMR', min_value=0.0, value=20000.0)

# Botón para predecir
if st.button('Predecir Precio'):
    # Crear un dataframe con los datos de entrada
    input_data = pd.DataFrame({
        'year': [year],
        'make': [make],
        'model': [model_name],
        'trim': [trim],
        'body': [body],
        'state': [state],
        'condition': [condition],
        'odometer': [odometer],
        'color': [color],
        'interior': [interior],
        'mmr': [mmr]
    })

    # Aplicar target encoding a las columnas categóricas
    categorical_columns = ['make', 'model', 'trim', 'body', 'state', 'color', 'interior']
    input_data[categorical_columns] = target_encoder.transform(input_data[categorical_columns])

    # Predecir el precio usando el modelo
    prediccion = model.predict(input_data)

    # Mostrar la predicción
    st.write(f"El precio estimado de venta es: ${prediccion[0]:,.2f}")