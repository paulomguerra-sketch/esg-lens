import streamlit as st
import pandas as pd

st.set_page_config(page_title="ESG Lens", layout="wide")

st.title("ESG Lens — Sustainability Report Analysis")

view = st.sidebar.radio("View", ["Company", "Comparison"])

if view == "Company":
    company = st.selectbox("Select company", ["Galp", "EDP", "Sonae", "Corticeira Amorim"])
    st.write(f"Showing data for **{company}**")

elif view == "Comparison":
    st.write("Cross-company comparison view")