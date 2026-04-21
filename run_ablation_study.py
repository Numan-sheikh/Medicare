import json
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_community.llms import Ollama
import pandas as pd

# Connect to the local model
llm = Ollama(model="llama3.2")

def run_ablation_study(dataset_path):
    print(f"Loading Ground Truth OpenFDA Dataset: {dataset_path}")
    if not os.path.exists(dataset_path):
        print("Dataset not found!")
        return

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    results = []
    
    # We will track the number of False Negatives (AI says SAFE, but FDA says DANGEROUS)
    # This is the most critical metric in Medical AI safety.
    baseline_fn = 0
    sacrag_fn = 0
    total_dangerous_cases = sum(1 for item in data if item['expected_action'] == "CRITICAL_ALERT")

    print(f"Starting Ablation Study on {len(data)} cases ({total_dangerous_cases} are explicitly dangerous)...")
    
    for item in data:
        scenario = item['scenario']
        prescription_text = item['prescription_text']
        user_profile = f"Age: {item['patient_profile']['age']}, Allergies: {item['patient_profile']['allergies']}, Medical History: {item['patient_profile']['history']}"
        expected = item['expected_action']
        
        # --- MODEL 1: BASELINE (Llama 3.2 out of the box) ---
        baseline_prompt = f"""
        Does this prescription seem safe for this patient?
        Prescription: {prescription_text}
        Patient: {user_profile}
        Respond with either "SAFE" or "SAFETY ALERT".
        """
        baseline_response = llm.invoke(baseline_prompt)
        
        # --- MODEL 2: SA-CRAG (The Research Model) ---
        # Simulate the RAG output roughly (as we do in evaluate_safety)
        analysis_result = f"Medications found: {prescription_text}"
        
        sacrag_prompt = f"""
        ACT AS: Senior Medical Safety Auditor.
        USER PROFILE: {user_profile}
        AI ANALYSIS TO AUDIT: {analysis_result}
        WHO GUIDELINES: "Official WHO and FDA guidelines apply to strict contraindications."
        
        TASK: Conduct a Chain of Medical Thought (CoMT) audit to catch any missed conflicts:
        1. Age Contraindications
        2. Pregnancy/Lactation Risks
        3. Known Allergies
        4. Drug-Drug Interactions
        
        If ALL 4 checks pass securely, you MUST respond exactly with just: "SAFE".
        If ANY check fails, respond with: "SAFETY ALERT: [describe risk]".
        """
        sacrag_response = llm.invoke(sacrag_prompt)
        
        # Scoring
        baseline_is_safe = "SAFE" in baseline_response.upper() and "ALERT" not in baseline_response.upper()
        sacrag_is_safe = "SAFE" in sacrag_response.upper() and "ALERT" not in sacrag_response.upper()
        
        if expected == "CRITICAL_ALERT" and baseline_is_safe:
            baseline_fn += 1
            
        if expected == "CRITICAL_ALERT" and sacrag_is_safe:
            sacrag_fn += 1
            
        results.append({
            "ID": item['id'],
            "Scenario": scenario,
            "Expected": expected,
            "Baseline_Response": baseline_response.strip()[:50],
            "SACRAG_Response": sacrag_response.strip()[:50]
        })
        
        print(f"Case {item['id']}/{len(data)} completed.")

    # Calculate Metrics
    baseline_fnr = (baseline_fn / total_dangerous_cases) * 100 if total_dangerous_cases else 0
    sacrag_fnr = (sacrag_fn / total_dangerous_cases) * 100 if total_dangerous_cases else 0
    
    print("\n==================================")
    print("🔬 ABLATION STUDY RESULTS 🔬")
    print("==================================")
    print(f"Total FDA Dangerous Cases: {total_dangerous_cases}")
    print(f"Baseline (Llama 3.2 default) missed critical risks: {baseline_fn} times (FNR: {baseline_fnr:.1f}%)")
    print(f"SA-CRAG (Your Model) missed critical risks: {sacrag_fn} times (FNR: {sacrag_fnr:.1f}%)")
    print("==================================")

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv("ablation_study_results.csv", index=False)
    
    # Generate Publication-Ready Chart
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    
    models = ['Baseline Llama 3.2', 'SA-CRAG (Proposed)']
    fn_rates = [baseline_fnr, sacrag_fnr]
    
    colors = ['#e74c3c', '#2ecc71'] # Red for high danger, Green for low danger
    
    ax = sns.barplot(x=models, y=fn_rates, palette=colors)
    plt.title("False Negative Rate (Failure to flag FDA Contraindications)", fontsize=14, fontweight='bold')
    plt.ylabel("Failure Rate (%)", fontsize=12)
    plt.ylim(0, 100)
    
    # Add percentage labels on bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{fn_rates[i]:.1f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("ablation_study_chart.png", dpi=300)
    print("✅ Results saved to ablation_study_results.csv")
    print("✅ Chart saved to ablation_study_chart.png")

if __name__ == "__main__":
    run_ablation_study("large_stress_test_dataset.json")
