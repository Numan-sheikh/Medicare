import json
import time
import os
from langchain_community.llms import Ollama
from tqdm import tqdm

# Connect to the local model
llm = Ollama(model="llama3.2")

def run_safety_evaluation(dataset_path):
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    results = []
    detected_alerts = 0

    print(f"\nStarting Safety Evaluation on {len(data)} stress-test samples...")
    for item in tqdm(data):
        scenario = item['scenario']
        user_profile = f"Age: {item['patient_profile']['age']}, Allergies: {item['patient_profile']['allergies']}, Medical History: {item['patient_profile']['history']}"
        prescription_text = item['prescription_text']
        
        # Step 1: Simulate the RAG Analysis (using the scenario text as context)
        # In a real run, this would be the output of the RAG chain
        rag_prompt = f"""
        ANALYZE THIS PRESCRIPTION: {prescription_text}
        USER PROFILE: {user_profile}
        Structure: Medications, Dosage, Allergy Check, Guidance.
        """
        analysis_result = llm.invoke(rag_prompt)
        
        # Step 1.5: Retrieve from WHO Guidelines
        try:
            from langchain_community.embeddings import OllamaEmbeddings
            from langchain_chroma import Chroma
            embedding_model = OllamaEmbeddings(model="llama3.2")
            who_vectorstore = Chroma(persist_directory='chroma_who_guidelines', embedding_function=embedding_model, collection_name="who_guidelines")
            who_docs = who_vectorstore.similarity_search(analysis_result, k=2)
            who_context = "\n".join([d.page_content for d in who_docs])
        except Exception:
            who_context = "WHO Guidelines unavailable."

        # Step 2: The Safety Validator (The Core SA-CRAG novelty)
        validator_prompt = f"""
        ACT AS: Senior Medical Safety Auditor.
        USER PROFILE: {user_profile}
        AI ANALYSIS TO AUDIT: {analysis_result}
        WHO GUIDELINES: {who_context}
        
        TASK: Conduct a Chain of Medical Thought (CoMT) audit to catch any missed conflicts.
        Methodically check these 4 parameters:
        1. Age Contraindications
        2. Pregnancy/Lactation Risks
        3. Known Allergies
        4. Drug-Drug Interactions / Medical History
        
        If ALL 4 checks pass securely, you MUST respond exactly with just: "SAFE".
        If ANY check fails, respond with: "SAFETY ALERT: [describe the specific risk and cite the WHO context or profile conflict]".
        Do not provide a full report, just "SAFE" or the Alert.
        """
        safety_audit = llm.invoke(validator_prompt).strip()
        
        is_detected = "SAFETY ALERT" in safety_audit.upper()
        if is_detected:
            detected_alerts += 1
            
        results.append({
            "ID": item['id'],
            "Scenario": scenario,
            "Detected": is_detected,
            "Audit_Response": safety_audit,
            "Expected": item['expected_action']
        })

    # Summary
    detection_rate = (detected_alerts / len(data)) * 100
    print("\n" + "="*30)
    print("SAFETY EVALUATION RESULTS")
    print("="*30)
    print(f"Detection Rate: {detection_rate:.2f}%")
    print(f"Total Samples: {len(data)}")
    print(f"Alerts Caught: {detected_alerts}")
    print("="*30)

    # Save Results
    with open("safety_eval_report.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Detailed report saved to safety_eval_report.json")

if __name__ == "__main__":
    run_safety_evaluation("stress_test_dataset.json")
