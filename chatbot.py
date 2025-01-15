import streamlit as st
import os

###############################################################################################
files=os.listdir()
###############################################################################################
files=[file for file in files if file.split(".")[-1]=="pdf"]

st.write(files)

st.title("Student Assistant")
