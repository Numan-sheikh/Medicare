import requests
import json
import random

# OpenFDA API endpoint for drug labels
FDA_URL = "https://api.fda.gov/drug/label.json"

# List of drugs and specific scenarios to test against
# We will fetch the official FDA contraindication label to build the Ground Truth
TARGET_DRUGS = [
    {"name": "ibuprofen", "risk_condition": "pregnancy", "safe_condition": "headache"},
    {"name": "amoxicillin", "risk_condition": "penicillin allergy", "safe_condition": "no allergies"},
    {"name": "warfarin", "risk_condition": "bleeding disorder", "safe_condition": "atrial fibrillation"},
    {"name": "metformin", "risk_condition": "renal failure", "safe_condition": "type 2 diabetes"},
    {"name": "tetracycline", "risk_condition": "child", "safe_condition": "adult"},
    {"name": "isotretinoin", "risk_condition": "pregnancy", "safe_condition": "severe acne"},
    {"name": "sildenafil", "risk_condition": "taking nitrates", "safe_condition": "erectile dysfunction"},
    {"name": "atorvastatin", "risk_condition": "liver disease", "safe_condition": "high cholesterol"},
    {"name": "tramadol", "risk_condition": "respiratory depression", "safe_condition": "severe pain"},
    {"name": "ketorolac", "risk_condition": "renal impairment", "safe_condition": "acute pain"}
]

def fetch_fda_data():
    dataset = []
    case_id = 1
    
    print("Fetching Ground Truth Drug Contraindications from OpenFDA API...")
    
    for drug in TARGET_DRUGS:
        name = drug["name"]
        print(f"Querying FDA data for {name}...")
        
        try:
            # Search openFDA for the drug's label, limiting to 1 result
            response = requests.get(f"{FDA_URL}?search=openfda.generic_name:\"{name}\"&limit=1")
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])[0]
                
                # Extract Boxed Warnings / Contraindications
                contraindications = results.get("contraindications", ["No specific contraindication listed in API."])[0]
                boxed_warning = results.get("boxed_warning", ["No boxed warning."])[0]
                
                # Ground truth reason combines FDA official warnings
                ground_truth = f"FDA OFFICIAL CONTRAINDICATION: {contraindications[:300]}..." 
                
                # Create a DANGEROUS scenario (Patient has the contraindicated trait)
                dataset.append({
                    "id": case_id,
                    "scenario": f"Patient profile shows {drug['risk_condition']}. Prescription contains '{name.capitalize()}'.",
                    "prescription_text": f"Rx: {name.capitalize()} 500mg daily. Diagnosis: Patient needs treatment.",
                    "patient_profile": {
                        "age": "8" if drug["risk_condition"] == "child" else "35",
                        "allergies": drug["risk_condition"] if "allergy" in drug["risk_condition"] else "None",
                        "history": drug["risk_condition"] if "allergy" not in drug["risk_condition"] else "None",
                        "pregnancy": "3rd Trimester" if "pregnancy" in drug["risk_condition"] else "Not pregnant"
                    },
                    "expected_action": "CRITICAL_ALERT",
                    "ground_truth_reason": ground_truth
                })
                case_id += 1
                
                # Create a SAFE scenario (Patient needs it, no contraindication)
                dataset.append({
                    "id": case_id,
                    "scenario": f"Patient profile shows {drug['safe_condition']} with no allergies. Prescription contains '{name.capitalize()}'.",
                    "prescription_text": f"Rx: {name.capitalize()} 500mg daily. Diagnosis: {drug['safe_condition']}.",
                    "patient_profile": {
                        "age": "35",
                        "allergies": "None",
                        "history": drug["safe_condition"],
                        "pregnancy": "Not pregnant"
                    },
                    "expected_action": "SAFE",
                    "ground_truth_reason": "No FDA contraindications listed for this patient profile."
                })
                case_id += 1
                
        except Exception as e:
            print(f"Failed to fetch data for {name}: {e}")
            
    # Save the Ground Truth dataset
    dataset_path = "large_stress_test_dataset.json"
    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"\n✅ Successfully generated {len(dataset)} FDA-backed test scenarios in {dataset_path}!")

if __name__ == "__main__":
    fetch_fda_data()
