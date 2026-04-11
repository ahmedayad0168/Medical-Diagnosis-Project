import streamlit as st
import pandas as pd
import joblib
import ollama
from fuzzywuzzy import process

st.set_page_config(page_title="Medical Support AI", page_icon="⚕️", layout="wide")

@st.cache_resource
def load_medical_assets():
    vectorizer = joblib.load("E:/my projects/MedicalDiagnosisProject/models/symptoms_models/vectorizer.pkl")
    rf_model = joblib.load("E:/my projects/MedicalDiagnosisProject/models/symptoms_models/medical_model.pkl")
    le = joblib.load("E:/my projects/MedicalDiagnosisProject/models/symptoms_models/label_encoder.pkl")
    data = pd.read_csv("E:/my projects/MedicalDiagnosisProject/data/mayo_clinic_all_diseases.csv")
    return vectorizer, rf_model, le, data

vectorizer, rf_model, le, data = load_medical_assets()

def predict_symptom(text):
    vec = vectorizer.transform([str(text).lower()])
    label_index = rf_model.predict(vec)[0]
    return le.inverse_transform([label_index])[0]

def lookup_disease_info(disease_query, info_type="smart_lookup"):
    unique_diseases = data['Disease'].dropna().unique().tolist()
    best_match, confidence = process.extractOne(disease_query, unique_diseases)
    
    if confidence < 75:
        return "Disease not found in database. Please check your spelling."
    
    match = data[data['Disease'] == best_match].iloc[0]
    
    if info_type == "get_causes":
        return f"### Causes for {best_match}\n\n{match['Causes']}"
    elif info_type == "get_overview":
        return f"### Overview of {best_match}\n\n{match['Overview']}"
    elif info_type == "get_risk_factors":
        return f"### Risk Factors for {best_match}\n\n{match['Risk Factors']}"
    else: 
        result = f"### Full Information: {best_match}\n"
        for col in data.columns:
            if col != "Disease" and pd.notna(match[col]):
                result += f"**{col}**: {match[col]}\n\n"
        return result

class MedicalSupportAgent:
    def __init__(self, model_name="llama3"):
        self.model_name = model_name

    def think(self, user_input):
        prompt = f"""You are a medical AI classifier. Categorize this user input: "{user_input}"
        Choose ONE keyword: symptom_prediction, get_causes, get_overview, get_risk_factors, smart_lookup.
        Respond with ONLY the keyword."""
        
        response = ollama.generate(model=self.model_name, prompt=prompt)
        return response['response'].strip().lower()

    def extract_disease(self, user_input):
        prompt = f"""Extract the disease name from: "{user_input}". If none, return NONE. 
        Respond with ONLY the name."""
        response = ollama.generate(model=self.model_name, prompt=prompt)
        name = response['response'].strip()
        return None if "NONE" in name.upper() else name

    def run(self, user_input):
        decision = self.think(user_input)
        
        if "symptom_prediction" in decision:
            predicted = predict_symptom(user_input)
            return f"Based on your symptoms, this could be: **{predicted}**."
        
        disease_name = self.extract_disease(user_input)
        if not disease_name:
            return "I couldn't identify a disease in your request. Please specify a condition."

        return lookup_disease_info(disease_name, decision)

st.title("⚕️ Agentic Medical Support Bot")
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Choose Ollama Model", ["llama3", "mistral", "phi3"], index=0)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you? (e.g., 'What are the causes of Pneumothorax?')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            try:
                agent = MedicalSupportAgent(model_name=model_choice)
                response = agent.run(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {str(e)}. Make sure Ollama is running and the model is pulled.")
