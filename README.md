This project takes user symptoms as input (in text form) and predicts the most likely disease using a trained machine learning model.

It demonstrates how AI and NLP can be applied in healthcare-related classification tasks.

---

## 🧠 How It Works

1. User inputs symptoms (e.g., fever, headache, cough)
2. Text is preprocessed and cleaned
3. Features are extracted using NLP techniques
4. Machine learning model predicts the disease
5. Output is returned to the user

---

## 🛠️ Technologies Used

* Python 🐍
* Pandas & NumPy
* Scikit-learn
* Natural Language Processing (TF-IDF / Embeddings)
* SQL Server (optional, for dataset storage)

---

## 📊 Model Pipeline

* Data Collection (Symptoms → Disease dataset)
* Data Preprocessing (cleaning & normalization)
* Feature Extraction (TF-IDF or embeddings)
* Model Training (classification algorithm)
* Prediction (real-time inference)

---

## 📁 Project Structure

```
Medical-Diagnosis-Project/
│
├── data/                  # Dataset files
├── models/               # Trained models
├── notebooks/            # Jupyter experiments
├── src/
│   ├── preprocess.py     # Data preprocessing
│   ├── train.py          # Model training
│   ├── predict.py        # Prediction script
│
├── app.py                # Optional UI/API
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/ahmedayad0168/Medical-Diagnosis-Project.git
cd Medical-Diagnosis-Project
```

### 2. Create virtual environment

```bash
python -m venv venv
```

Activate it:

* Windows:

```bash
venv\Scripts\activate
```

* Mac/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the prediction script:

```bash
python src/predict.py
```

### Example Input:

```
fever, cough, sore throat
```

### Example Output:

```
Predicted Disease: Flu
```

---

## 📈 Features

* Symptom-based disease prediction
* NLP-based text processing
* Machine learning classification model
* Easy to extend with new datasets
* Can be integrated with web apps (Flask / Streamlit)

---

## 🚀 Future Improvements

* Improve model accuracy with deep learning (BERT / Transformers)
* Add web interface (Streamlit / Flask)
* Add explanation system (why prediction was made)
* Expand dataset for more diseases
* Integrate medical knowledge base

---
## datasets 
```bash
https://www.mayoclinic.org/
```
