import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About Project")

st.header("Overview")
st.write("""
The Waste Classification System is an AI-powered application that automatically classifies 
waste items to promote sustainable waste management and recycling.
""")

st.header("Problem")
st.write("- Improper waste segregation leads to contamination")
st.write("- Manual sorting is time-consuming and error-prone")
st.write("- Limited public awareness about proper recycling")
st.write("- Waste facilities lack automation")

st.header("Solution")
st.write("- Automated classification using deep learning")
st.write("- Real-time processing and instant results")
st.write("- Educational guidance for users")
st.write("- Easy-to-use interface")

st.caption("♻️ Waste Classification System")
