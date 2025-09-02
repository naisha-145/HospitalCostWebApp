import streamlit as st
import pandas as pd
import joblib
import openai

# ==========================
# Load your ML model
# ==========================
model = joblib.load("hospital_cost_model.pkl")

# ==========================
# OpenAI API Key
# ==========================
# Go to https://platform.openai.com/account/api-keys and create a key
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"  
openai.api_key = OPENAI_API_KEY

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="Hospital Cost Predictor & Chatbot", layout="wide")
st.title("🏥 Hospital Cost Predictor & AI Chatbot")

# Split page into two columns
col1, col2 = st.columns(2)

# ==========================
# Column 1: Hospital Cost Predictor
# ==========================
with col1:
    st.header("Predict Hospital Cost")
    
    # Collect patient info
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    smoker = st.selectbox("Smoker", ["Yes", "No"])
    department = st.text_input("Department", "General")
    proc_mri = st.checkbox("Procedure: MRI")
    proc_blood_test = st.checkbox("Procedure: Blood Test")
    com_diabetes = st.checkbox("Comorbidity: Diabetes")
    com_hypertension = st.checkbox("Comorbidity: Hypertension")
    
    # Prepare input
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
    
    if st.button("Predict Cost"):
        try:
            pred = float(model.predict(input_df)[0])
            st.success(f"Predicted Hospital Cost: ₹{pred:,.2f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ==========================
# Column 2: RAG-style AI Chatbot
# ==========================
with col2:
    st.header("Ask about hospital or health topics")
    
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    user_input = st.text_input("You:", key="input")

    if st.button("Send") and user_input.strip() != "":
        with st.spinner("Generating answer..."):
            # Call OpenAI GPT
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful hospital assistant."},
                        *[
                            {"role": "user", "content": chat['user']}
                            for chat in st.session_state['chat_history']
                        ],
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=250
                )
                answer = response['choices'][0]['message']['content']
                st.session_state.chat_history.append({"user": user_input, "bot": answer})
            except Exception as e:
                st.error(f"Chatbot error: {e}")

    # Display chat history
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")

