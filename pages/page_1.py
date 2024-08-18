import streamlit as st
import pandas as pd
import numpy as np
import wget
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

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

st.write("Predicted values:", predictions)
st.write("Mean Squared Error:", mse)
st.write("R-squared Score:", r2)
