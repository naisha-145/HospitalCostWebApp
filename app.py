import streamlit as st
import pandas as pd
import joblib
import openai

# ---------------------------
# 1. Load your ML model
# ---------------------------
@st.cache_data
def load_model():
    return joblib.load("hospital_cost_model.pkl")

model = load_model()

# ---------------------------
# 2. OpenAI API setup
# ---------------------------
openai.api_key = "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"  # <-- Replace this with your key

def generate_answer(question):
    """Simple RAG chatbot using OpenAI GPT"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful hospital assistant. Answer questions about hospital procedures, insurance, and health."},
            {"role": "user", "content": question}
        ],
        temperature=0.7,
        max_tokens=200
    )
    return response['choices'][0]['message']['content']

# ---------------------------
# 3. Streamlit UI
# ---------------------------
st.set_page_config(page_title="Hospital Cost Predictor & Chatbot", layout="wide")
st.title("🏥 Hospital Cost Predictor & AI Chatbot")

col1, col2 = st.columns(2)

# --- Column 1: Cost Prediction ---
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
            predicted_cost = float(model.predict(input_df)[0])
            st.success(f"💰 Predicted Hospital Cost: ₹ {predicted_cost:,.2f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# --- Column 2: AI Chatbot ---
with col2:
    st.header("Ask our AI Chatbot")
    
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    user_input = st.text_input("Ask a question about hospital procedures, insurance, or health:")
    
    if st.button("Send") and user_input.strip() != "":
        with st.spinner("Generating answer..."):
            answer = generate_answer(user_input)
            st.session_state.chat_history.append({"user": user_input, "bot": answer})
    
    # Display chat history
    for chat in st.session_state.get('chat_history', []):
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
