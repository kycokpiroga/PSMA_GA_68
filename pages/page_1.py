import streamlit as st
import pandas as pd
import numpy as np
import wget
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import datetime
def func_page_1():
    st.set_page_config(
        page_title="PSMA",
        page_icon="👋",
    )
    st.header('DOTA-PSMA-617_Ga-68_2', divider='rainbow')

def preparation():
    psma_617_df = pd.read_csv('PSMA.csv')
    output = psma_617_df['A']
    features = psma_617_df[['day_from_calib_gen', 'k_rec']]
    return features, output

def ml_polynomial(features, output):
    degree = 4
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    # Fit the model
    model.fit(features, output)
    
    # Predict
    self_pred = model.predict(features)
    
    # Calculate scores
    mse = mean_squared_error(output, self_pred)
    r2 = r2_score(output, self_pred)
    
    return self_pred, mse, r2

# Prepare the data
features, output = preparation()

# Perform polynomial regression
predictions, mse, r2 = ml_polynomial(features, output)

st.write("Mean Squared Error:", mse)
st.write("R-squared Score:", r2)
st.header('Внесите данные, Дата калибровки генератора галлия, дата и время предыдущего синтеза или ТЭ, дата и время предполагаемого синтеза', divider='rainbow')
col1, col2, col3 = st.columns(3)
with col1:
    # Ввод даты
    date = st.date_input("Выберите дату из паспорта генератора", datetime.date.today())

with col2:
     # Ввод даты
    date = st.date_input("Выберите дату и время передачи на КК последнего синтеза или ТЭ", datetime.date.today())

    # Ввод времени в формате строки
    time_str = st.text_input("Введите время", "12:00")

    # Преобразование строки в объект времени
    try:
        time = datetime.datetime.strptime(time_str, "%H:%M").time()
    except ValueError:
        st.error("Неверный формат времени. Пожалуйста, используйте формат HH:MM.")
    # Объединение даты и времени
    datetime_selected = datetime.datetime.combine(date, time)

with col3:
     # Ввод даты
    date_2 = st.date_input("Выберите дату и время передачи на КК предполагаемого синтеза", datetime.date.today())

    # Ввод времени в формате строки
    time_str_2 = st.text_input("Введите время", "13:00")

    # Преобразование строки в объект времени
    try:
        time_2 = datetime.datetime.strptime(time_str_2, "%H:%M").time()
    except ValueError:
        st.error("Неверный формат времени. Пожалуйста, используйте формат HH:MM.")


    # Объединение даты и времени
    datetime_selected_2 = datetime.datetime.combine(date_2, time_2)

