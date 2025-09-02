import streamlit as st
import joblib
import pandas as pd
import openai  # Blackbox AI-compatible

# ----------------- STREAMLIT PAGE SETTINGS -----------------
st.set_page_config(page_title="Hospital AI Assistant", layout="wide")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🏥 Hospital AI Assistant 🏥</h1>", unsafe_allow_html=True)

# ----------------- LOAD ML MODEL -----------------
@st.cache_resource
def load_model():
    return joblib.load("hospital_cost_model.pkl")

model = load_model()

# ----------------- RAG CHATBOT -----------------
openai.api_key = "sk-zV46P0zFQ9AYqna2JfYzog"  # Replace with your Blackbox AI API key

def generate_answer(question):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": question}]
    )
    return response['choices'][0]['message']['content']

# ----------------- TWO COLUMN LAYOUT -----------------
col1, col2 = st.columns(2)

# ----------------- COLUMN 1: HOSPITAL COST PREDICTOR -----------------
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
            pred = float(model.predict(input_df)[0])
            st.success(f"Predicted Hospital Cost: ₹{pred:,.2f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ----------------- COLUMN 2: AI CHATBOT -----------------
with col2:
    st.header("Ask the AI Chatbot")
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    user_input = st.text_input("Ask anything about hospital procedures or health:")

    if st.button("Send") and user_input.strip() != "":
        with st.spinner("Generating answer..."):
            answer = generate_answer(user_input)
            st.session_state.chat_history.append({"user": user_input, "bot": answer})

    # Display chat history
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
