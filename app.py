# app.py
import streamlit as st
import pandas as pd
import joblib
import os
import openai

# -------------------------
# Load your ML model
# -------------------------
@st.cache_data(show_spinner=True)
def load_model():
    return joblib.load("hospital_cost_model.pkl")  # make sure this file is in your repo

model = load_model()

# -------------------------
# OpenAI API setup
# -------------------------
# Make sure you have set the secret in Streamlit Cloud:
# OPENAI_API_KEY="your_openai_key_here"
openai.api_key = st.secrets["OPENAI_API_KEY"]

# -------------------------
# RAG Chatbot function
# -------------------------
def generate_answer(user_query):
    """
    Basic RAG-style chatbot using OpenAI GPT.
    For real RAG, you can combine a vector store & retrieval. 
    Here, we keep it simple for hackathon purposes.
    """
    prompt = f"You are a helpful hospital assistant. Answer this question concisely:\n{user_query}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Hospital Cost Predictor + AI Chatbot", layout="wide")
st.title("🏥 Hospital Cost Predictor & AI Chatbot")

# Create two columns: left for ML prediction, right for chatbot
col1, col2 = st.columns(2)

# -------------------------
# Column 1: Hospital Cost Prediction
# -------------------------
with col1:
    st.header("Predict Hospital Cost")
    
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    smoker = st.selectbox("Smoker", ["Yes", "No"])
    department = st.text_input("Department", "General")
    
    proc_mri = st.checkbox("Procedure: MRI")
    proc_blood_test = st.checkbox("Procedure: Blood Test")
    
    com_diabetes = st.checkbox("Comorbidity: Diabetes")
    com_hypertension = st.checkbox("Comorbidity: Hypertension")
    
    if st.button("Predict Cost"):
        input_dict = {
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "smoker": smoker,
            "department": department,
            "proc_mri": int(proc_mri),
            "proc_blood_test": int(proc_blood_test),
            "com_diabetes": int(com_diabetes),
            "com_hypertension": int(com_hypertension)
        }
        input_df = pd.DataFrame([input_dict])
        
        try:
            predicted_cost = float(model.predict(input_df)[0])
            st.success(f"💰 Predicted Hospital Cost: ₹{predicted_cost:,.2f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# -------------------------
# Column 2: RAG-based AI Chatbot
# -------------------------
with col2:
    st.header("Ask the AI Chatbot")
    
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    user_input = st.text_input("Ask a question about hospital procedures, insurance, or health:")
    
    if st.button("Send") and user_input.strip() != "":
        with st.spinner("Generating answer..."):
            answer = generate_answer(user_input)
            st.session_state.chat_history.append({"user": user_input, "bot": answer})
    
    # Display chat history
    for chat in st.session_state['chat_history']:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
