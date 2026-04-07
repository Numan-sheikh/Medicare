import json
import time
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import string

# ==================== Metric Functions ====================

def normalize_answer(s):
    """Normalize answer text for robust matching."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, ground_truth):
    """Exact Match (EM) logic."""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1(prediction, ground_truth):
    """F1 Score for token overlap."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return int(prediction_tokens == ground_truth_tokens)
    
    common = set(prediction_tokens) & set(ground_truth_tokens)
    num_same = len(common)
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

class SemanticEvaluator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading local embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
    
    def get_similarity(self, s1, s2):
        embeddings = self.model.encode([s1, s2])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(sim)

# ==================== Evaluation Engine ====================

def run_evaluation(dataset_path, rag_pipeline_func):
    """
    RAG Evaluation Engine.
    dataset_path: Path to JSON/CSV with 'question' and 'ground_truth_answer'
    rag_pipeline_func: A lambda or function that takes a question and returns a predicted answer
    """
    # Load dataset
    if dataset_path.endswith('.json'):
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    else:
        data = pd.read_csv(dataset_path).to_dict('records')

    results = []
    semantic_eval = SemanticEvaluator()

    print(f"\nStarting Evaluation on {len(data)} samples...")
    for idx, item in enumerate(tqdm(data)):
        question = item['question']
        ground_truth = item['ground_truth_answer']

        # Measure Latency
        start_time = time.time()
        try:
            prediction = rag_pipeline_func(question)
        except Exception as e:
            print(f"Error at question '{question}': {e}")
            prediction = "ERROR"
        latency = time.time() - start_time

        # Calculate Metrics
        em_score = compute_exact_match(prediction, ground_truth)
        f1_score = compute_f1(prediction, ground_truth)
        semantic_sim = semantic_eval.get_similarity(prediction, ground_truth) if prediction != "ERROR" else 0.0

        # Heuristic Error Categorization (Prompting User if needed, or based on scores)
        error_type = "Correct"
        if em_score < 1.0:
            if semantic_sim > 0.8:
                error_type = "Partial Correctness"
            elif semantic_sim < 0.4:
                error_type = "Retrieval Failure / Irrelevant"
            else:
                error_type = "Generation Hallucination / Deviation"

        results.append({
            "Question": question,
            "Ground Truth": ground_truth,
            "Prediction": prediction,
            "EM": em_score,
            "F1": f1_score,
            "Semantic Similarity": semantic_sim,
            "Latency (s)": latency,
            "Error Category": error_type
        })

    # Summary Statistics
    results_df = pd.DataFrame(results)
    summary = {
        "Mean EM": results_df["EM"].mean(),
        "Mean F1": results_df["F1"].mean(),
        "Mean Semantic Sim": results_df["Semantic Similarity"].mean(),
        "Mean Latency (s)": results_df["Latency (s)"].mean(),
        "Total Error Count": len(results_df[results_df["Error Category"] != "Correct"])
    }

    print("\n" + "="*30)
    print("EVALUATION RESULTS SUMMARY")
    print("="*30)
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")
    
    print("\n" + "-"*30)
    print("ERROR DISTRIBUTION")
    print("-"*30)
    error_counts = results_df["Error Category"].value_counts()
    for category, count in error_counts.items():
        print(f"{category}: {count}")
    print("="*30)

    # Save to CSV
    output_file = "evaluation_report.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")
    
    return results_df

# ==================== User Plug-in Here ====================

if __name__ == "__main__":
    # --- STEP 1: Connect to Real RAG Pipeline ---
    try:
        from langchain_chroma import Chroma
        from langchain_community.llms import Ollama
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain.prompts import PromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser

        # Initialize Models (Matching app.py)
        llm = Ollama(model="llama3.2")
        embedding_model = OllamaEmbeddings(model="llama3.2")
        
        # Load Existing Vector Store
        DB_PATH = 'chroma_medical_db'
        if not os.path.exists(DB_PATH):
            print(f"Error: {DB_PATH} not found. Please upload a PDF in the Medicare app first.")
            exit(1)

        vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embedding_model,
            collection_name="medical_docs"
        )
        
        # Replicate RAG Prompt from app.py
        rag_prompt_template = """
        SYSTEM PERSONA: You are Medicare, a prescription analysis expert.
        USER DETAILS: {user_details}
        PRESCRIPTION CONTEXT: {context}
        YOUR TASK: Analyze the provided prescription context based on the user's details. Structure your response with these exact markdown headings:
        **Overall Safety Assessment:**
        **Dosage Check:**
        **Allergy & Interaction Check:**
        **Guidance:**
        """
        rag_prompt = PromptTemplate.from_template(rag_prompt_template)
        
        # Define the Standard User Profile
        user_profile = "Age: 35, Allergies: None, Medical History: No chronic conditions"

        # Build the Chain
        rag_chain = (
            {"context": vectorstore.as_retriever(), "user_details": RunnablePassthrough()}
            | rag_prompt | llm | StrOutputParser()
        )

        def my_rag_pipeline(question):
            """True RAG pipeline using Medicare documents."""
            return rag_chain.invoke(user_profile + " " + question)

        # --- STEP 2: Run Evaluation ---
        sample_dataset = "eval_dataset.json"
        if not os.path.exists(sample_dataset):
            print(f"Error: Dataset {sample_dataset} not found. Please ensure it exists.")
        else:
            run_evaluation(sample_dataset, my_rag_pipeline)

    except ImportError as e:
        print(f"Error: Missing library for True RAG. Please run: pip install langchain-chroma langchain-community langchain-core")
        print(f"Original error: {e}")

