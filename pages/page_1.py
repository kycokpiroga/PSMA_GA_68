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

def user_u():
    col1, col2, col3 = st.columns(3)
    with col1:
        # Ввод даты
        date_0 = st.date_input("Выберите дату из паспорта генератора", datetime.date.today())

    with col2:
        # Ввод даты
        date_1 = st.date_input("Выберите дату и время передачи на КК последнего синтеза или ТЭ", datetime.date.today())

        # Ввод времени в формате строки
        time_str = st.text_input("Введите время", "07:30")

        # Преобразование строки в объект времени
        try:
            time_1 = datetime.datetime.strptime(time_str, "%H:%M").time()
        except ValueError:
            st.error("Неверный формат времени. Пожалуйста, используйте формат HH:MM.")
        # Объединение даты и времени
        datetime_selected_1 = datetime.datetime.combine(date_1, time_1)

    with col3:
        # Ввод даты
        date_2 = st.date_input("Выберите дату и время передачи на КК предполагаемого синтеза", datetime.date.today())

        # Ввод времени в формате строки
        time_str_2 = st.text_input("Введите время", "11:30")

        # Преобразование строки в объект времени
        try:
            time_2 = datetime.datetime.strptime(time_str_2, "%H:%M").time()
        except ValueError:
            st.error("Неверный формат времени. Пожалуйста, используйте формат HH:MM.")

        # Объединение даты и времени
        datetime_selected_2 = datetime.datetime.combine(date_2, time_2)

    day_from_calib_gen = (date_2 - date_0).days
    st.write("Количество дней между датой калибровки и предполгаемым синтезом:", day_from_calib_gen)
    
    k_rec = (datetime_selected_2 - datetime_selected_1).total_seconds() / 60
    st.write("Количество минут между синтезами:", k_rec)
    st.write("Количество часов между синтезами:", round(k_rec / 60, 2))
    return day_from_calib_gen, k_rec

# Prepare the data
features, output = preparation()

# Perform polynomial regression
predictions, mse, r2 = ml_polynomial(features, output)

st.write("Mean Squared Error:", mse)
st.write("R-squared Score:", r2)
st.header('Внесите данные, Дата калибровки генератора галлия, дата и время предыдущего синтеза или ТЭ, дата и время предполагаемого синтеза', divider='rainbow')

# Call the user_u function to display the input fields and get the values
day_from_calib_gen, k_rec = user_u()

def prep_syntes(k_rec, day_from_calib_gen):
    k_rec = -0.000008 * (k_rec ** 2) + 0.00515 * k_rec + 0.1555
    day_from_calib_gen = day_from_calib_gen * (-1)
    
    if k_rec < 0:
        k_rec = 1
    if k_rec > 1:
        k_rec = 1
    return k_rec, day_from_calib_gen

# Call the predict_syntes function with the values from user_u
k_rec, day_from_calib_gen = prep_syntes(k_rec, day_from_calib_gen)

# Display the results
st.write("k_rec:", k_rec)
st.write("day_from_calib_gen:", day_from_calib_gen)

def syn_predict():
    syn_pred = model.predict(features)