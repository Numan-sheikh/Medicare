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
Install all required libraries, including the Core RAG framework, Machine Learning packages, and Evaluation tools:
```powershell
# Core Project Libraries
pip install -r requirements.txt

# Machine Learning & Manual Dependencies
pip install tensorflow pandas werkzeug==2.3.7
conda install -c conda-forge keras opencv -y
conda install -c anaconda scikit-learn scikit-image flask -y
pip install fpdf flask-sqlalchemy jupyter

# NEW: RAG Evaluation Dependencies
pip install sentence-transformers tqdm langchain-chroma langchain-community langchain-core
```

### 3. Setup Ollama and Llama 3.2
Ensure Ollama is running, then pull the required model:
```powershell
ollama pull llama3.2
```

---

## 📊 Evaluation System

Medicare now includes a robust evaluation suite to measure the performance of the RAG pipeline.

### Running the Evaluation
1.  **Prepare the Dataset**: Ensure `eval_dataset.json` is present in the root directory.
2.  **Activate Environment**: `conda activate medical_chatbot`
3.  **Run Evaluation**:
    ```powershell
    python evaluate_rag.py
    ```

### Metrics Measured:
- **Exact Match (EM)**: Percent of answers that are identical to ground truth.
- **F1 Score**: Token-level overlap between predicted and true answers.
- **Semantic Similarity**: Vector-based meaning comparison (Local Sentence-Transformers).
- **Latency**: Average time taken for a full RAG cycle.
- **Error Analysis**: Automatic categorization of Hallucinations vs. Retrieval Failures.

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
