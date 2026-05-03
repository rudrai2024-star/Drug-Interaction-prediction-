🚀 Drug Interaction Prediction System (DDI Predictor Pro)
📌 Project Overview
This project presents an advanced Drug–Drug Interaction (DDI) Prediction System that analyzes multiple medications using machine learning, cheminformatics, and AI-based clinical reasoning.
Drug–drug interactions are a major concern in healthcare, especially in cases of polypharmacy, where multiple drugs are prescribed together. This system aims to predict, explain, and visualize potential interactions in a user-friendly and interpretable way.
The application supports both:
Manual drug selection
Automated prescription-based analysis
🎯 Objectives
Predict interaction risks between multiple drugs
Provide clinically meaningful explanations
Identify affected organs and visualize impact
Reduce manual effort using prescription analysis
Improve interpretability using explainable AI
⚙️ Core Features
🔬 1. Multi-Drug Interaction Prediction
Supports 2–6 drugs simultaneously
Generates all pairwise combinations
Uses probability-based prediction
⚠️ 2. Contraindication Detection
Identifies life-threatening drug combinations
Stops pipeline immediately if detected
🧠 3. Machine Learning Model
Random Forest classifier
4098-dimensional feature vector
Fast and efficient real-time prediction
🧪 4. Molecular Feature Engineering
SMILES → Morgan Fingerprints (2048-bit)
Captures structural drug properties
🤖 5. AI Clinical Insights (Groq API)
Side effects generation
Mechanism of interaction
Organ system detection
🫀 6. Organ Impact Analysis
Calculates organ-wise risk scores
Highlights most affected systems
🧍 7. Interactive Body Map
Color-coded organ visualization
Clickable organ insights
Cumulative & maximum scoring modes
📄 8. Prescription Analysis
Upload PDF or image
Extract drugs using LLM
Auto-run prediction
📊 9. Visualization Dashboard
Risk cards
Gauge charts
Molecular structures
🧠 10. SHAP Explainability
Feature contribution analysis
Local prediction explanation
🕘 11. Session History
Stores previous analyses
Tracks risk over time
🏗️ System Workflow
User Input / Prescription Upload
            ↓
Contraindication Check
            ↓
Feature Engineering (RDKit)
            ↓
ML Prediction (Random Forest)
            ↓
Groq LLM Analysis
            ↓
Organ Impact Scoring
            ↓
Visualization Dashboard + Body Map
⚙️ Tech Stack
🔹 Frontend
Streamlit
HTML / CSS
Plotly
🔹 Backend
Python
🔹 Machine Learning
scikit-learn (Random Forest)
🔹 Cheminformatics
RDKit
🔹 AI Integration
Groq API (LLM-based insights)
🔹 Data Processing
NumPy
Pandas
🔹 Model Handling
Joblib
🧠 Model Details
Model: Random Forest
Input:
2048-bit fingerprint (Drug A)
2048-bit fingerprint (Drug B)
Dosage A + Dosage B
Total: 4098 features
📈 Performance:
AUC-ROC ≈ 0.93
Recall ≈ 90%
Brier Score ≈ 0.097
📂 Project Structure
Drug-Interaction-Prediction/
│
├── app.py
├── models/
├── data/
├── utils/
├── assets/
├── notebooks/
├── requirements.txt
└── README.md
🚀 Installation
1. Clone Repository
git clone https://github.com/rudrai2024-star/Drug-Interaction-prediction-
cd Drug-Interaction-prediction-
2. Install Dependencies
pip install -r requirements.txt
3. Add Groq API Key
Create:
.streamlit/secrets.toml
Add:
GROQ_API_KEY = "your_api_key_here"
4. Run Application
streamlit run app.py
🧪 Example Usage
Select multiple drugs
Enter dosage values
Click Analyze
View:
Interaction probability
Risk level
Side effects
Organ impact
⚠️ Disclaimer
This system is intended for educational and research purposes only.
It should not be used as a substitute for professional medical advice.
🚧 Limitations
Limited dataset coverage
Dependency on external API (Groq)
Fingerprint-based (not graph-based)
🔮 Future Scope
Graph Neural Networks (GNN)
DrugBank full integration
3D body visualization (Three.js)
Clinical database integration
Advanced optimization techniques
👨‍💻 Author
Rudrajyoti Paul
Garima Ola
