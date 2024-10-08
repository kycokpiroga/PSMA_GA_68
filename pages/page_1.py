import streamlit as st
import pandas as pd
import numpy as np
import wget
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import datetime

def func_page_1():
    st.set_page_config(
        page_title="PSMA",
        page_icon="👋",
    )
    st.header('DOTA-PSMA-617_Ga-68_2', divider='rainbow')

#def preparation():
    #psma_617_df = pd.read_csv('PSMA.csv')
    #output = psma_617_df['A']
    #features = psma_617_df[['day_from_calib_gen', 'k_rec']]
    #st.scatter_chart(psma_617_df, x="day_from_calib_gen", y="A")
    #st.subheader('Визуальное предствление используемой модели')
    #return features, output

#def ml_gradient_boosting(features, output): # требуется если только upd модель
    #model = GradientBoostingRegressor()

    # Fit the model
    #model.fit(features, output)
    
    # Predict
    #self_pred = model.predict(features)
    
    # Calculate scores
    #mse = mean_squared_error(output, self_pred)
    #r2 = r2_score(output, self_pred)
    #st.write(mse, r2)
    #return model, self_pred, mse, r2

#def save_model(model):
    #with open('gradient_boosting_psma.pkl', 'wb') as psma_pickle:
        #pickle.dump(model, psma_pickle)

def load_model():
    with open('gradient_boosting_psma.pkl', 'rb') as psma_pickle:
        model = pickle.load(psma_pickle)
    return model

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
    st.write("Количество дней между датой калибровки генератора и предполагаемым синтезом:", day_from_calib_gen)
    
    k_rec = (datetime_selected_2 - datetime_selected_1).total_seconds() / 60
    st.write("Количество минут между синтезами:", k_rec)
    st.write("Количество часов между синтезами:", round(k_rec / 60, 2))
    return day_from_calib_gen, k_rec

# Prepare the data
#features, output = preparation()

# Check if the model file exists, if not, train and save the model
#try:
    #model = load_model()
#except FileNotFoundError:
    #model, self_pred, mse, r2 = ml_gradient_boosting(features, output)
    #save_model(model)

st.caption('Внесите данные, Дата калибровки генератора Галлия-68, дата и время предыдущего синтеза или ТЭ, дата и время предполагаемого синтеза')
# Call the user_u function to display the input fields and get the values
day_from_calib_gen, k_rec = user_u()

def prep_syntes(k_rec, day_from_calib_gen):
    k_rec = -0.000008 * (k_rec ** 2) + 0.0052 * k_rec + 0.1552
    day_from_calib_gen = day_from_calib_gen * (-1)
    
    if k_rec < 0:
        k_rec = 1
    if k_rec > 1:
        k_rec = 1
    return k_rec, day_from_calib_gen

# Call the predict_syntes function with the values from user_u
k_rec, day_from_calib_gen = prep_syntes(k_rec, day_from_calib_gen)

# Display the results
st.write("k_rec:", round(k_rec,3))
#st.write("day_from_calib_gen:", day_from_calib_gen)

# Создание DataFrame с переменными
data = {'day_from_calib_gen': [day_from_calib_gen],
    'k_rec': [k_rec]  
}
# Load the model
model = load_model()

user_data = pd.DataFrame(data)
# Вывод DataFrame
#st.write(user_data)
user_pred = model.predict(user_data)
user_pred = user_pred.round(-1)
st.write("Предполагаемое значение активности при передачи в КК МБк:")
st.success(user_pred)
st.success("± 100 МБк,  R²  = 0.90")
