#imports 
import streamlit as st
from predict_page import show_predict_page
from interpret_page import show_interpret_page

#dropdown tw switch between predict and interpret page 
page = st.sidebar.selectbox("Predict Or Interpret", ("Predict", "Interpret"))
if page == "Predict":
    show_predict_page()
else: 
    show_interpret_page()



