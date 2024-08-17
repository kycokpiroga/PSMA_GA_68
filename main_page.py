import streamlit as st
import pandas as pd
import numpy as np
from pages.page_1 import func_page_1
def main():
    st.sidebar.subheader('Page selection')
    page_selection = st.sidebar.selectbox('Please select a page',['Welcome',
    'Second'])
    pages_main = {
    'Welcome': main_page,
    'Second': run_page_1.py}
    # Run selected page
    pages_main[page_selection]()
def main_page():
        st.set_page_config(
        page_title="Welcome",
        page_icon="👋",)
        st.header('DOTA-PSMA-617_Ga-68', divider='rainbow')
        #st.image('phone.png')
        st.markdown("#### Моделирование активности препарта в МБк")
        st.divider()
        st.markdown ("- ### Цели и ожидания пользователей")
        st.markdown ("- ### Возможные риски и ограничения модели")  
        st.divider()
        st.button("Reset", type="primary")
def run_page_1():
        func_page_1()
if __name__ == "__main__":   
    main()