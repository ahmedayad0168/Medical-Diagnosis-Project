import streamlit as st
import sys
sys.path.append(".")
from src.models.symptom_predictor import SymptomPredictor
from src.models.lookup import DiseaseLookup
from src.agent.medical_agent import MedicalSupportAgent

@st.cache_resource
def load_components():
    # Adjust paths to your saved models
    symptom = SymptomPredictor(
        vectorizer_path="models/symptom_model/vectorizer.pkl",
        model_path="models/symptom_model/classifier.pkl",
        le_path="models/symptom_model/label_encoder.pkl"
    )
    lookup = DiseaseLookup("data/processed/mayo_clinic_all_diseases.csv")
    agent = MedicalSupportAgent(symptom, lookup, ollama_model="llama3")
    return agent

st.set_page_config(page_title="MedQuery", page_icon="⚕️")
st.title("🏥 MedQuery – Your Medical Q&A Assistant")

agent = load_components()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask something (symptoms, causes, overview, risk factors)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.run(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})