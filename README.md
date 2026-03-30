# 🩺 Medicare: AI-Powered Virtual First Doctor

**Medicare** is an intelligent medical chatbot designed to provide instant, safe, and reliable healthcare guidance. It leverages the power of **Large Language Models (LLMs)** and **Retrieval-Augmented Generation (RAG)** to analyze symptoms, predict possible illnesses, and provide safety checks for prescriptions.

---

## 🚀 Features

- **AI Disease Prediction**: Guided symptom-to-diagnosis conversations.
- **Intelligent Prescription Analysis**: RAG-supported safety checks for drug interactions and allergies.
- **Personalized Patient Profiles**: Tailored advice based on age, medical history, and allergies.
- **PDF Report Generation**: Downloadable reports for your personal records or professional consultation.
- **Modern UI**: A premium, "Neural Glass" aesthetic for a seamless experience.

---

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:
1. **Anaconda**: [Download here](https://www.anaconda.com/products/individual)
2. **Ollama**: [Download here](https://ollama.com/) (Required for the Llama 3.2 model)

---

## 📦 Installation & Setup

Follow these steps to set up the project on your local machine:

### 1. Create the Conda Environment
Open your **Anaconda Prompt** and run:
```powershell
conda create -n medical_chatbot python=3.10 -y
conda activate medical_chatbot
```

### 2. Install Dependencies
Install all required libraries, including the Core RAG framework and Machine Learning packages:
```powershell
# Core Project Libraries
pip install -r requirements.txt

# Machine Learning & Manual Dependencies
pip install tensorflow pandas werkzeug==2.3.7
conda install -c conda-forge keras opencv -y
conda install -c anaconda scikit-learn scikit-image flask -y
pip install fpdf flask-sqlalchemy jupyter
```

### 3. Setup Ollama and Llama 3.2
Ensure Ollama is running, then pull the required model:
```powershell
ollama pull llama3.2
```

---

## 🏃 Running the Application

1. **Activate the Environment**:
    ```powershell
    conda activate medical_chatbot
    ```

2. **Start the Flask Server**:
    ```powershell
    python app.py
    ```

3. **Access the App**:
    Open your browser and navigate to:
    **[http://127.0.0.1:5001](http://127.0.0.1:5001)**

---

## 📂 Project Structure

- `app.py`: Main Flask application logic and RAG chain setup.
- `templates/`: HTML front-end files.
- `static/`: CSS, JS, and image assets.
- `requirements.txt`: List of core project dependencies.
- `instance/`: Contains the SQLite database (`users.db`).
- `uploads/`: Temporary storage for uploaded prescription PDFs.

---

## ⚠️ Disclaimer
*Medicare is an AI-powered assistant and is **not** a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.*

---
**Developed with ❤️ for better healthcare accessibility.**
