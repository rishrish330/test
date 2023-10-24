import streamlit as st
from impressions import predict

st.title("Impressions Predictor")
query = st.text_input("Enter your query")
a = predict(query)
# if 'status' in a:
#   st.write(a["status"])
if "Top Hashtags" in a:
  st.write("High Performing Hashtags")
  st.write(a['Top Hashtags'])
