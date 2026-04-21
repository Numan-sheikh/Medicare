import os
import shutil
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, Response
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import json
from fpdf import FPDF

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
except ImportError:
    analyzer = None
    anonymizer = None

def anonymize_text(text):
    if not analyzer or not anonymizer:
        return text # Fallback if presidio isn't installed properly
    results = analyzer.analyze(text=text, entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION", "CREDIT_CARD", "SSN"], language='en')
    anonymized_result = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized_result.text

# Load environment variables
load_dotenv()

# ==================== Flask App Setup ====================
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a-default-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ==================== RAG and LLM Setup ====================
UPLOAD_FOLDER = 'uploads'
DB_PATH = 'chroma_medical_db'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)
llm = Ollama(model="llama3.2")
embedding_model = OllamaEmbeddings(model="llama3.2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
vectorstore = None
rag_chain = None

# ==================== Database Model ====================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    allergies = db.Column(db.String(300), nullable=True)
    medical_history = db.Column(db.String(500), nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# ==================== Helper Functions ====================
def create_pdf_report(title, content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(0, 10, content.encode('latin-1', 'replace').decode('latin-1'))
    return Response(
        pdf.output(dest='S').encode('latin-1'),
        mimetype='application/pdf',
        headers={'Content-Disposition': 'attachment;filename=Medicare_Report.pdf'}
    )

def setup_rag_chain_from_file(file_path):
    global vectorstore, rag_chain
    try:
        # Clear existing vectorstore from memory if it exists to avoid connection locks
        vectorstore = None
        
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
            
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # CHECK: Was any text actually extracted? (Detect scanned images or empty files)
        if not documents:
            raise ValueError("No readable text found in the PDF. Please ensure your prescription is not a low-quality scan or photo.")
            
        # PII Redaction Step
        print("Scrubbing PII from prescription...")
        for doc in documents:
            doc.page_content = anonymize_text(doc.page_content)
            
        docs = text_splitter.split_documents(documents)
        
        # Double check split results
        if not docs:
            raise ValueError("The PDF content was found but could not be processed into chunks.")

        # Initialize Chroma with explicit persistence settings
        vectorstore = Chroma.from_documents(
            documents=docs, 
            embedding=embedding_model, 
            persist_directory=DB_PATH,
            collection_name="medical_docs" # Using a named collection for stability
        )
        rag_prompt_template = """
        SYSTEM PERSONA: You are Medicare, a professional medical prescription analysis expert. 
        USER CLINICAL PROFILE: {user_details}
        PRESCRIPTION TEXT (Retrieved Context): {context}
        
        TASK: Conduct a rigorous analysis of the prescription. 
        1. Identify the medications and their dosages.
        2. Cross-reference with the User Clinical Profile (Allergies, History, Age).
        3. Flag any potential contradictions or safety risks.
        
        Structure your response with these exact markdown headings:
        **1. Medications Found:**
        **2. Dosage & Safety Assessment:**
        **3. Allergy & Interaction Check:**
        **4. Clinical Guidance:**
        
        IMPORTANT: If the prescription text is unclear or if there is a direct conflict with the user's allergies, you MUST highlight this in the Safety Assessment.
        """
        rag_prompt = PromptTemplate.from_template(rag_prompt_template)
        rag_chain = (
            {"context": vectorstore.as_retriever(), "user_details": RunnablePassthrough()}
            | rag_prompt | llm | StrOutputParser()
        )
        return True
    except Exception as e:
        print(f"Error setting up RAG chain: {e}")
        return False

# ==================== Routes ====================



@app.route('/')
def root():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

# NEW: Route for the home page
@app.route('/home')
def home():
    if 'user_id' in session:
        return render_template('home.html')
    return redirect(url_for('login'))

@app.route('/features')
def features():
    if 'user_id' in session:
        return render_template('features.html')
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash('Username and password are required.')
            return redirect(url_for('register'))
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists.')
            return redirect(url_for('register'))
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.')
    return redirect(url_for('login'))


@app.route('/assistant')
def assistant():
    if 'user_id' in session:
        session['chat_stage'] = 'GENERAL'
        return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/chat', methods=['POST'])
def chat():
    if 'user_id' not in session: return jsonify({'error': 'Unauthorized'}), 401
    data = request.json
    user_input = data.get('message')
    is_prediction_start = data.get('is_prediction_start', False)
    user = db.session.get(User, session['user_id'])
    user_profile = f"Age: {user.age or 'Not provided'}, Allergies: {user.allergies or 'Not provided'}, Medical History: {user.medical_history or 'Not provided'}"
    chat_stage = session.get('chat_stage', 'GENERAL')

    if is_prediction_start:
        chat_stage = 'PREDICTION_STARTED'
        session['symptom_data'] = {'initial': user_input, 'answers': []}

    try:
        ai_response = ""
        if chat_stage == 'PREDICTION_STARTED':
            prompt = f"""SYSTEM PERSONA: You are a medical data gathering AI. A user has reported these symptoms: "{session['symptom_data']['initial']}". Generate exactly 3 concise follow-up questions to clarify the condition. Your output MUST be a JSON array of strings, like ["question 1", "question 2", "question 3"]. Do not output any other text."""
            response_text = llm.invoke(prompt)
            questions = json.loads(response_text)
            session['follow_up_questions'] = questions
            session['chat_stage'] = 'AWAITING_ANSWER_1'
            ai_response = questions[0]
        elif chat_stage == 'AWAITING_ANSWER_1':
            session['symptom_data']['answers'].append(user_input)
            session['chat_stage'] = 'AWAITING_ANSWER_2'
            ai_response = session['follow_up_questions'][1]
        elif chat_stage == 'AWAITING_ANSWER_2':
            session['symptom_data']['answers'].append(user_input)
            session['chat_stage'] = 'AWAITING_ANSWER_3'
            ai_response = session['follow_up_questions'][2]
        elif chat_stage == 'AWAITING_ANSWER_3':
            session['symptom_data']['answers'].append(user_input)
            s_data = session['symptom_data']
            full_context = f"Initial Symptoms: {s_data['initial']}. Answers: 1. {s_data['answers'][0]}, 2. {s_data['answers'][1]}, 3. {s_data['answers'][2]}"
            
            # THIS IS THE CORRECT, FULL PROMPT
            prompt = f"""
            SYSTEM PERSONA: You are Medicare, a virtual first doctor.
            USER DATA: Profile: {user_profile}. Full Symptom Report: {full_context}.
            YOUR TASK: Based on all user data, generate a structured medical report with these exact markdown headings:
            **Disclaimer:** (Warn that you are an AI and not a substitute for a real doctor.)
            **Possible Illness(es):** (List 1-2 likely conditions.)
            **Recommended Generic Medicines:** (Suggest over-the-counter medicines like Paracetamol with dosage.)
            **Lifestyle and Home Care:** (Provide a bulleted list of advice.)
            **When to See a Doctor:** (List critical symptoms that require immediate medical attention.)
            """
            ai_response = llm.invoke(prompt)
            session['last_prediction_result'] = ai_response
            session.pop('last_analysis_result', None)
            session['chat_stage'] = 'GENERAL'
            session.pop('symptom_data', None)
            session.pop('follow_up_questions', None)
        else: # GENERAL chat stage
            # TRIAGE ROUTER (Guardrail)
            triage_prompt = f"Is the following input related to healthcare, medical advice, or inquiring about a medical prescription?\nInput: '{user_input}'\nReply EXACTLY with the word 'NO' ONLY if it is completely unrelated (like coding, math, or general knowledge). Reply 'YES' if it relates to the patient's health or their uploaded files."
            triage_result = llm.invoke(triage_prompt).strip().upper()
            
            # If the model explicitly says NO, block it.
            if triage_result.startswith("NO") or " NO " in f" {triage_result} ":
                return jsonify({'response': "I am Medicare, a clinical AI assistant. I am strictly designed for healthcare queries and cannot answer non-medical questions. How can I assist you with your health today?"})
                
            prompt = f"You are Medicare, a helpful medical assistant. A user with profile ({user_profile}) asks: '{user_input}'. Answer informatively."
            ai_response = llm.invoke(prompt)
        return jsonify({'response': ai_response})
    except Exception as e:
        print(f"Error in /chat route: {e}")
        session['chat_stage'] = 'GENERAL'
        return jsonify({'error': 'An error occurred. Please try again.'}), 500

@app.route('/get_user_info', methods=['GET'])
def get_user_info():
    if 'user_id' not in session: return jsonify({'error': 'Unauthorized'}), 401
    user = db.session.get(User, session['user_id'])
    if user:
        return jsonify({'age': user.age or '', 'allergies': user.allergies or '', 'medical_history': user.medical_history or ''})
    return jsonify({'error': 'User not found'}), 404

@app.route('/update_user_info', methods=['POST'])
def update_user_info():
    if 'user_id' not in session: return jsonify({'error': 'Unauthorized'}), 401
    user = db.session.get(User, session['user_id'])
    if user:
        data = request.json
        user.age = data.get('age') or user.age
        user.allergies = data.get('allergies') or user.allergies
        user.medical_history = data.get('medical_history') or user.medical_history
        db.session.commit()
        return jsonify({'message': 'Profile updated successfully.'})
    return jsonify({'error': 'User not found'}), 404

@app.route('/analyze_prescription', methods=['POST'])
def analyze_prescription():
    if 'user_id' not in session: return jsonify({'error': 'Unauthorized'}), 401
    if 'file' not in request.files: return jsonify({'error': 'No prescription file provided.'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No file selected.'}), 400
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        if not setup_rag_chain_from_file(file_path):
            return jsonify({'error': 'Failed to process the prescription PDF.'}), 500
        user = db.session.get(User, session['user_id'])
        user_details = f"Age: {user.age}, Allergies: {user.allergies}, Medical History: {user.medical_history}"
        
        # Step 1: Initial RAG Analysis
        analysis_result = rag_chain.invoke(user_details)
        
        # Step 1.5: Retrieve from WHO Guidelines (Ground Truth)
        try:
            who_vectorstore = Chroma(persist_directory='chroma_who_guidelines', embedding_function=embedding_model, collection_name="who_guidelines")
            who_docs = who_vectorstore.similarity_search(analysis_result, k=2)
            who_context = "\n".join([f"[Page {d.metadata.get('page', 'Unknown')}] {d.page_content}" for d in who_docs])
        except Exception:
            who_context = "WHO Guidelines unavailable."

        # Step 2: SA-CRAG CoMT Safety Validator Loop (The Research Novelty)
        validator_prompt = f"""
        ACT AS: Senior Medical Safety Auditor.
        USER PROFILE: {user_details}
        AI ANALYSIS TO AUDIT: {analysis_result}
        WHO GUIDELINES: {who_context}
        
        TASK: Conduct a Chain of Medical Thought (CoMT) audit to catch any missed conflicts.
        Methodically check these 4 parameters:
        1. Age Contraindications
        2. Pregnancy/Lactation Risks
        3. Known Allergies
        4. Drug-Drug Interactions / Medical History
        
        If ALL 4 checks pass securely, you MUST respond exactly with just: "SAFE".
        If ANY check fails, respond with: "SAFETY ALERT: [describe risk and YOU MUST prominently cite the exact WHO [Page X] or conflict from context]".
        Do not provide a full report, just "SAFE" or the Alert.
        """
        safety_audit = llm.invoke(validator_prompt).strip()
        
        final_response = analysis_result
        if "SAFETY ALERT" in safety_audit.upper():
            final_response = f"### ⚠️ CRITICAL SAFETY WARNING\n**{safety_audit}**\n\n---\n" + analysis_result
            
        session['last_analysis_result'] = final_response
        session.pop('last_prediction_result', None)
        return jsonify({'response': final_response})
    except ValueError as ve:
        print(f"Validation error setting up RAG: {ve}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"Prescription analysis error: {e}")
        return jsonify({'error': f'An error occurred during analysis: {str(e)}'}), 500

@app.route('/download_prediction_pdf')
def download_prediction_pdf():
    if 'user_id' not in session: return "Unauthorized", 401
    result_text = session.get('last_prediction_result')
    if not result_text: return "No prediction result found to download.", 404
    return create_pdf_report("Medicare - Disease Prediction Report", result_text)

@app.route('/download_analysis_pdf')
def download_analysis_pdf():
    if 'user_id' not in session: return "Unauthorized", 401
    result_text = session.get('last_analysis_result')
    if not result_text: return "No analysis result found to download.", 404
    return create_pdf_report("Medicare - Prescription Analysis Report", result_text)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5001, use_reloader=False)