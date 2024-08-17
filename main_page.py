import streamlit as st
import pandas as pd
import numpy as np
from pages.page_1 import func_page_1
def main():
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
if __name__ == "__main__":   
    main()