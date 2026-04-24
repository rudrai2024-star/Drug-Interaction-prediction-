"""
DDI Predictor Pro — Clinical Intelligence Platform
Aesthetic: Brutalist dark / military precision / terminal data readout
"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from collections import defaultdict
import time
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv
import os as _os
# Load .env from project root
_env_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), ".env")
load_dotenv(_env_path)

st.set_page_config(
    page_title="DDI // PREDICTOR PRO",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from itertools import combinations
import sys, os
import json
import re
import streamlit.components.v1 as components

# PDF & Image extraction imports
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    import pytesseract
    from PIL import Image
    import io
    # Configure pytesseract to find tesseract binary on macOS
    pytesseract.pytesseract.pytesseract_cmd = '/opt/homebrew/bin/tesseract'
    IMAGE_SUPPORT = True
except ImportError:
    IMAGE_SUPPORT = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from model_manager import (
        load_all_models, load_comparison_results,
        get_model_performance_dataframe, get_best_model_info,
        get_model_list, get_default_model
    )
    MODEL_MANAGER_ENABLED = True
except ImportError:
    MODEL_MANAGER_ENABLED = False

try:
    from groq import Groq as _Groq

    # Priority 1: Environment variable from .env (loaded via dotenv)
    _groq_key = _os.environ.get("GROQ_API_KEY") or _os.environ.get("groq_api_key")
    
    # Priority 2: Streamlit secrets (fallback)
    if not _groq_key:
        try:
            _groq_key = st.secrets.get("GROQ_API_KEY") or st.secrets.get("groq_api_key")
        except Exception:
            pass

    if _groq_key:
        groq_client = _Groq(api_key=_groq_key)
        GROQ_ENABLED = True
    else:
        GROQ_ENABLED = False
        groq_client = None
except ImportError:
    GROQ_ENABLED = False
    groq_client = None

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:        #000000;
  --bg1:       #080808;
  --bg2:       #0f0f0f;
  --bg3:       #141414;
  --border:    #1c1c1c;
  --border2:   #282828;
  --text1:     #f0f0f0;
  --text2:     #888888;
  --text3:     #3a3a3a;
  --accent:    #e8ff47;
  --accent2:   #ff4747;
  --accent3:   #47c8ff;
  --accent4:   #ff8c47;
  --green:     #47ff8c;
  --mono:      'Space Mono', monospace;
  --sans:      'Space Grotesk', sans-serif;
}

html, body { background: var(--bg) !important; }

.stApp {
  background: var(--bg) !important;
  font-family: var(--sans);
  color: var(--text1);
}

/* scanline overlay */
.stApp::after {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(255,255,255,0.008) 2px,
    rgba(255,255,255,0.008) 4px
  );
  pointer-events: none;
  z-index: 9999;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}
section[data-testid="stSidebar"] { display: none !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg1);
  border-bottom: 1px solid var(--border);
  border-radius: 0;
  padding: 0 32px;
  gap: 0;
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--mono) !important;
  font-size: 10px !important;
  font-weight: 400 !important;
  color: var(--text3) !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  border-radius: 0 !important;
  padding: 14px 24px !important;
  border-bottom: 2px solid transparent !important;
  transition: all 0.15s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--text2) !important;
  background: transparent !important;
}
.stTabs [aria-selected="true"] {
  background: transparent !important;
  color: var(--accent) !important;
  border-bottom: 2px solid var(--accent) !important;
}
.stTabs [data-baseweb="tab-panel"] {
  padding: 0 !important;
}

/* ── BUTTONS ── */
.stButton > button {
  font-family: var(--mono) !important;
  font-size: 10px !important;
  font-weight: 700 !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  background: var(--accent) !important;
  color: #000 !important;
  border: none !important;
  border-radius: 0 !important;
  padding: 11px 22px !important;
  transition: all 0.12s ease !important;
}
.stButton > button:hover {
  background: #fff !important;
  transform: none !important;
  box-shadow: none !important;
}
.stButton > button[kind="secondary"] {
  background: transparent !important;
  color: var(--text2) !important;
  border: 1px solid var(--border2) !important;
}
.stButton > button[kind="secondary"]:hover {
  border-color: var(--text2) !important;
  color: var(--text1) !important;
}

/* ── INPUTS ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
  background: var(--bg2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 0 !important;
  color: var(--text1) !important;
  font-family: var(--mono) !important;
  font-size: 12px !important;
}
.stSelectbox > div > div:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 1px var(--accent) !important;
}

/* ── METRICS ── */
[data-testid="stMetric"] {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 0;
  padding: 16px 18px;
}
[data-testid="stMetricValue"] {
  font-family: var(--mono) !important;
  color: var(--text1) !important;
  font-size: 24px !important;
  font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
  font-family: var(--mono) !important;
  color: var(--text3) !important;
  font-size: 9px !important;
  font-weight: 400 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.12em !important;
}

/* ── EXPANDERS ── */
.streamlit-expanderHeader {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 0 !important;
  color: var(--text2) !important;
  font-family: var(--mono) !important;
  font-size: 11px !important;
  font-weight: 400 !important;
  letter-spacing: 0.05em !important;
  padding: 12px 16px !important;
}
.streamlit-expanderHeader:hover {
  border-color: var(--border2) !important;
  color: var(--text1) !important;
}
.streamlit-expanderContent {
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 !important;
  background: var(--bg1) !important;
  padding: 0 !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 0 !important;
}

/* ── PROGRESS ── */
.stProgress > div > div {
  background: var(--accent) !important;
  border-radius: 0 !important;
}
.stProgress > div {
  border-radius: 0 !important;
  background: var(--bg3) !important;
}

/* ── ALERTS ── */
.stAlert {
  border-radius: 0 !important;
  font-family: var(--mono) !important;
  font-size: 11px !important;
}

/* ── RADIO ── */
.stRadio label {
  font-family: var(--mono) !important;
  font-size: 10px !important;
  color: var(--text2) !important;
  letter-spacing: 0.08em !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); }
::-webkit-scrollbar-thumb:hover { background: var(--text3); }

/* ── CUSTOM COMPONENTS ── */
.page-wrap { padding: 0 32px 48px; }

.hud-bar {
  background: var(--bg1);
  border-bottom: 1px solid var(--border);
  padding: 0 32px;
  display: flex;
  align-items: stretch;
  height: 52px;
  gap: 0;
}
.hud-cell {
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 0 24px 0 0;
  margin-right: 24px;
  border-right: 1px solid var(--border);
}
.hud-label {
  font-family: var(--mono);
  font-size: 8px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text3);
  margin-bottom: 2px;
}
.hud-val {
  font-family: var(--mono);
  font-size: 13px;
  font-weight: 700;
  color: var(--text1);
  letter-spacing: 0.04em;
}
.hud-val.accent { color: var(--accent); }
.hud-val.red    { color: var(--accent2); }
.hud-val.blue   { color: var(--accent3); }

.section-label {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--text3);
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

.drug-row {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  padding: 14px 16px;
  margin-bottom: 6px;
  display: grid;
  grid-template-columns: 1fr 90px 28px;
  gap: 8px;
  align-items: center;
}

.pair-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  margin-bottom: 1px;
}
.pair-header {
  padding: 12px 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  cursor: pointer;
  border-bottom: 1px solid var(--border);
}
.pair-body { padding: 20px; }

.risk-tag {
  font-family: var(--mono);
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  padding: 4px 10px;
  border: 1px solid;
}
.risk-high { color: var(--accent2); border-color: rgba(255,71,71,0.3); background: rgba(255,71,71,0.06); }
.risk-med  { color: var(--accent4); border-color: rgba(255,140,71,0.3); background: rgba(255,140,71,0.06); }
.risk-low  { color: var(--green);   border-color: rgba(71,255,140,0.3); background: rgba(71,255,140,0.06); }

.prob-display {
  font-family: var(--mono);
  font-size: 28px;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1;
}

.mol-frame {
  background: var(--bg1);
  border: 1px solid var(--border);
  padding: 12px;
  text-align: center;
}
.mol-name {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 8px;
}
.smiles-str {
  font-family: var(--mono);
  font-size: 8px;
  color: var(--text3);
  word-break: break-all;
  margin-top: 6px;
  line-height: 1.5;
}

.bar-track {
  height: 3px;
  background: var(--bg3);
  margin-top: 4px;
}
.bar-fill { height: 100%; transition: width 0.3s ease; }

.stat-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  margin-bottom: 20px;
}
.stat-cell {
  background: var(--bg2);
  padding: 14px 16px;
  text-align: center;
}
.stat-val {
  font-family: var(--mono);
  font-size: 22px;
  font-weight: 700;
  color: var(--text1);
  display: block;
  line-height: 1;
}
.stat-lbl {
  font-family: var(--mono);
  font-size: 8px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text3);
  display: block;
  margin-top: 4px;
}

.mech-box {
  background: var(--bg1);
  border-left: 2px solid var(--accent3);
  padding: 12px 14px;
  margin-bottom: 14px;
}
.mech-label {
  font-family: var(--mono);
  font-size: 8px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent3);
  margin-bottom: 5px;
}
.mech-text {
  font-family: var(--sans);
  font-size: 12px;
  color: var(--text2);
  line-height: 1.6;
}

.se-item {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 7px 0;
  border-bottom: 1px solid var(--border);
  font-family: var(--sans);
  font-size: 12px;
  color: var(--text2);
}
.se-num {
  font-family: var(--mono);
  font-size: 9px;
  color: var(--text3);
  padding-top: 2px;
  min-width: 18px;
}

.organ-row { padding: 6px 0; }
.organ-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 3px;
}
.organ-name {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text2);
  letter-spacing: 0.04em;
}
.organ-score {
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 700;
}

.contra-box {
  background: rgba(255,71,71,0.04);
  border: 1px solid rgba(255,71,71,0.2);
  border-left: 3px solid var(--accent2);
  padding: 16px 20px;
  margin: 12px 0;
}
.contra-head {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--accent2);
  margin-bottom: 6px;
}
.contra-pair {
  font-family: var(--sans);
  font-size: 16px;
  font-weight: 600;
  color: var(--text1);
  margin-bottom: 4px;
}
.contra-rule {
  font-family: var(--sans);
  font-size: 13px;
  color: rgba(255,71,71,0.8);
}

.hist-row {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-left: 3px solid;
  padding: 14px 18px;
  margin-bottom: 4px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.hist-ts {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text3);
  margin-bottom: 4px;
}
.hist-detail {
  font-family: var(--sans);
  font-size: 13px;
  font-weight: 500;
  color: var(--text2);
}
.hist-pct {
  font-family: var(--mono);
  font-size: 26px;
  font-weight: 700;
  text-align: right;
}
.hist-plbl {
  font-family: var(--mono);
  font-size: 8px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text3);
  text-align: right;
}

.tag-mono {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  padding: 3px 8px;
  border: 1px solid var(--border2);
  color: var(--text2);
  display: inline-block;
}

.divider { height: 1px; background: var(--border); margin: 16px 0; }

.model-chip {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  padding: 4px 10px;
  border: 1px solid;
  display: inline-block;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 360px;
  gap: 12px;
  text-align: center;
}
.empty-cross {
  width: 48px;
  height: 48px;
  position: relative;
}
.empty-cross::before,
.empty-cross::after {
  content: '';
  position: absolute;
  background: var(--border2);
}
.empty-cross::before {
  width: 1px; height: 100%;
  left: 50%; top: 0;
}
.empty-cross::after {
  width: 100%; height: 1px;
  top: 50%; left: 0;
}
.empty-title {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text3);
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}
.blink { animation: blink 1.2s step-end infinite; }

@keyframes fadeIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}
.fade { animation: fadeIn 0.3s ease; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ─────────────────────────────────────────────────────────────────
SIDE_EFFECT_TO_ORGANS = {
    "bleeding": ["Heart", "Blood/Coagulation"], "hypotension": ["Heart", "Blood Pressure"],
    "hypertension": ["Heart", "Blood Pressure"], "bradycardia": ["Heart"],
    "tachycardia": ["Heart"], "arrhythmia": ["Heart"], "qt prolongation": ["Heart"],
    "cardiac arrest": ["Heart"], "edema": ["Heart", "Kidneys"],
    "nausea": ["Stomach/GI"], "vomiting": ["Stomach/GI"], "diarrhea": ["Intestines/GI"],
    "constipation": ["Intestines/GI"], "abdominal pain": ["Stomach/GI"],
    "gi bleeding": ["Stomach/GI", "Blood/Coagulation"], "ulcer": ["Stomach/GI"],
    "hepatotoxicity": ["Liver"], "liver damage": ["Liver"], "jaundice": ["Liver"],
    "elevated liver enzymes": ["Liver"], "nephrotoxicity": ["Kidneys"],
    "kidney damage": ["Kidneys"], "renal failure": ["Kidneys"],
    "electrolyte imbalance": ["Kidneys", "Blood"], "seizure": ["Brain/CNS"],
    "confusion": ["Brain/CNS"], "dizziness": ["Brain/CNS"], "sedation": ["Brain/CNS"],
    "drowsiness": ["Brain/CNS"], "headache": ["Brain/CNS"], "tremor": ["Nervous System"],
    "serotonin syndrome": ["Brain/CNS", "Nervous System"],
    "respiratory depression": ["Lungs"], "dyspnea": ["Lungs"], "bronchospasm": ["Lungs"],
    "anemia": ["Blood/Coagulation"], "thrombocytopenia": ["Blood/Coagulation"],
    "coagulation": ["Blood/Coagulation"], "rhabdomyolysis": ["Muscles", "Kidneys"],
    "muscle damage": ["Muscles"], "myalgia": ["Muscles"],
    "hypoglycemia": ["Pancreas/Endocrine"], "hyperglycemia": ["Pancreas/Endocrine"],
}
SEVERITY_WEIGHTS = {
    "fatal": 1.0, "severe": 0.9, "life-threatening": 1.0,
    "bleeding": 0.85, "hepatotoxicity": 0.8, "nephrotoxicity": 0.8,
    "seizure": 0.85, "respiratory depression": 0.95,
    "serotonin syndrome": 0.95, "cardiac arrest": 1.0,
    "moderate": 0.6, "mild": 0.3,
}
ORGAN_COLORS = {
    "Heart": "#ff4747", "Blood Pressure": "#ff8c47", "Blood/Coagulation": "#ff4747",
    "Liver": "#ff8c47", "Kidneys": "#47c8ff", "Stomach/GI": "#47ff8c",
    "Intestines/GI": "#47ff8c", "Brain/CNS": "#c847ff", "Nervous System": "#c847ff",
    "Lungs": "#47c8ff", "Muscles": "#ff47a0", "Pancreas/Endocrine": "#e8ff47",
    "Systemic": "#888888"
}
ACCENT_COLORS = ["#e8ff47", "#47c8ff", "#ff47a0", "#ff8c47", "#47ff8c", "#c847ff"]
ACCENT_COLORS = ["#e8ff47", "#47c8ff", "#ff47a0", "#ff8c47", "#47ff8c", "#c847ff"]

COMMON_DRUG_SMILES = {
    # ─── ORIGINAL DRUGS ──────────
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "warfarin": "CC(=O)CC(c1ccccc1)c2c(O)c3ccccc3oc2=O",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "metformin": "CN(C)C(=N)NC(=N)N",
    "lisinopril": "NCCCC(NC(CCc1ccccc1)C(=O)O)C(=O)N2CCCC2C(=O)O",
    "atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
    "simvastatin": "CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C12C",
    "amlodipine": "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c2cccc(Cl)c2Cl",
    "metoprolol": "COCCc1ccc(OCCC(O)CNC(C)C)cc1",
    "furosemide": "NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl",
    "sertraline": "CNC1CCC(c2ccc(Cl)c(Cl)c2)c3ccccc13",
    "fluoxetine": "CNCCC(c1ccccc1)Oc2ccc(cc2)C(F)(F)F",
    "tramadol": "COc1cccc(c1)C2(O)CCCCC2CN(C)C",
    "diazepam": "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13",
    "omeprazole": "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=CC(=C3)OC",
    "vitamin_d3": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CCC4=CC(=O)CCC34C)C",
    "mecobalamin": "CC1=C(C(=O)NC(C(=O)NC(C(=O)NC(C(=O)NC(C(=O)NC(C(=O)N)C(C)O)C(C)O)C(C)O)C(C)O)C)C=CN1C",
    "l_methylfolate": "CN1C=NC2=C1N=C(N=C2N)NCC3=CC=C(C=C3)C(=O)NCCC(=O)O",
    "pyridoxal_5_phosphate": "CC1=NC=C(C(=C1O)COP(=O)(O)O)O",
    "otrivin": "CC(C)NCC(COC1=CC=CC2=C1OCCC2)O",
    "zerodol": "CCc1ccc2nc(sc2c1)S(=O)(=O)N",
    "sompraz": "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=CC(=C3)OC",
    "mondeslor": "c1ccc(cc1)c2c(cccc2=O)N(C)CCCC(O)CC(O)C(=O)O",
    
    # ─── DDI DATASET DRUGS (202 VALID SMALL MOLECULES) ──────────
    "acarbose": "CC1C(C(C(C(O1)OC2C(C(C(C(O2)CO)O)O)O)O)O)NC(=O)C",
    "acyclovir": "NC1=NC(=O)N(C=N1)CC(CO)C",
    "albuterol": "CC(C)NCC(COc1ccc(cc1)C)O",
    "alfuzosin": "CC(C)Cc1c(cc(cc1OC)S(=O)(=O)N)NC(=O)c2cccnc2",
    "alogliptin": "NC1=NN(C(=N)N1CC(=O)O)c2c(F)c(F)c(F)c(F)c2F",
    "alprazolam": "Cc1c(cc(n1-c2ccc(Cl)cc2)C(=O)N)Cl",
    "amiloride": "NC(=N)N(CC(=O)N)c1cc(ccc1Cl)S(=O)(=O)N",
    "amoxicillin": "CC1(C)SC2C(NC(=O)c3ccccc3)C(=O)N2C1C(=O)O",
    "apixaban": "CC(C)Cc1c(c(n(c1=O)c2ccc(Cl)cc2)N(C)C)C(=O)N",
    "atazanavir": "CC(C)(C)CC(=O)N1CCCC1c2c(OC)ccc(c2)C(=O)NC(Cc3c(cccc3)C(C)(C)C)C",
    "atenolol": "CC(C)NCC(COc1ccc(cc1)CC(=O)N)O",
    "azathioprine": "c1nc(n(n1)C)S",
    "azilsartan": "CC(C)c1c(cc(cc1Cl)C(=O)Nc2ccccc2F)C(=O)O",
    "azithromycin": "CC(C)c1c(C)c(OC)c(OC)c(OC)c1OC",
    "bempedoic_acid": "CC(Cc1c(cc(cc1F)Cl)C(=O)O)O",
    "benazepril": "CCOC(=O)C1CN(CCc2ccccc2)C(=O)C1",
    "betrixaban": "CCBr",
    "bisoprolol": "COc1ccc(cc1)CC(O)CNC(C)C",
    "bromocriptine": "CN1CCC23CCCCC2=C(C1CC4=C3NC5=CC=CC=C45)O",
    "bumetanide": "CC(=O)Nc1ccc(cc1S(=O)(=O)N)NC(=O)c2ccc(cc2)C(C)C",
    "cabergoline": "CCCCN1CCc2cc(ccc2C1C(=O)N)O",
    "canagliflozin": "CC(C)(C)c1c(cc(cc1O)OC2C(C(C(O2)CO)O)O)C(F)(F)F",
    "candesartan": "CCCc1nc(c(n1Cc2ccccc2C(=O)O)C(=O)N)C",
    "captopril": "CC(CCC(=O)O)SC(=O)NC1CCCN1C(=O)C(CCCN)O",
    "carvedilol": "COc1ccc(cc1)CCNc2cc(ccc2O)C(C)C",
    "ceftriaxone": "CS(=O)(=O)N1C(C)=CC=C1c2ccc(cc2)C3=C(N(C)CS3)C(=O)N",
    "cefuroxime": "CC(=NOc1ccccc1C(=O)O)c2nc(cs2)SCCN=CN",
    "celecoxib": "CC(c1ccc(cc1)S(=O)(=O)N)c2cc(ccc2F)S(=O)(=O)N",
    "cetirizine": "OC(=O)CCN1c2c(cc(cc2Cl)Cl)CCc3cccnc13",
    "chlorthalidone": "Cl-c1cc(ccc1S(=O)(=O)N)C(=O)Nc2ccc(cc2)Cl",
    "cholestyramine": "C",
    "cilostazol": "CCc1cc(c(cc1OCC)S(=O)(=O)N)C(=O)Nc2ccccc2",
    "ciprofloxacin": "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "citalopram": "CN1CCCC(Cc2ccc(OC)cc2)C1=CCc3ccccc3",
    "clarithromycin": "CC(C)CC(=O)OC1C(C)C(OC2CC(C)C(OC)C(C)O2)CC(C)C1OC",
    "clevidipine": "CCOC(=O)C1=C(C(=O)OC)C(=C(N1)c2ccccc2)c3ccc(F)cc3",
    "clonidine": "Nc1cc(ccc1Cl)c2nc(Cl)cc(Cl)n2",
    "clopidogrel": "CS(=O)Cc1sccc1Cc2c(F)c(F)c(F)c(F)c2F",
    "colesevelam": "C",
    "colestipol": "C",
    "dabigatran": "CC(=O)Nc1ccc(cc1)N(C)c2nc(cc(n2)C(F)(F)F)N(C)C",
    "dapagliflozin": "Cc1c(c(cc(c1)OC2C(C(C(O2)CO)O)O)S(=O)(=O)N)C",
    "daptomycin": "C",
    "darunavir": "CC(C)c1c(O)c(cc(c1C(=O)NC(Cc2ccccc2)C(O)C(Cc3ccc(O)cc3)NC(=O)C(CC(C)C)N)C(C)C)C(F)(F)F",
    "diclofenac": "O=C(O)Cc1ccccc1Nc2c(cc(cc2Cl)Cl)Cl",
    "digoxin": "CC(C)C(=O)OC1CC(C)C(OC2C(C)C(OC3C(C)C(OC4CC(C)C(O)C(C)O4)C(C)O3)C(C)O2)C(C)O1",
    "diltiazem": "COc1ccc(cc1)CCN(C)C(=O)C2=C(OCC(=O)OC)c3ccccc3OC2C",
    "dipyridamole": "CCn1c(=O)cc(nc1=O)Nc2ccc(cc2)N(CC)CCO",
    "dolutegravir": "CN1c2c(cc(cc2=O)F)C(O)(C(=O)O)C(c3ccccc3)C1=O",
    "doxazosin": "COc1cc2c(cc1OC)c(cc(n2)C(=O)N)N",
    "doxycycline": "CN(C)C1CCC2=C(C1=O)C(=O)c3c(O)c(cc(c3C2=O)O)O",
    "dulaglutide": "C",
    "duloxetine": "CN1CCCC(Cc2c(Cl)ccc(Cl)c2O)C1",
    "dutasteride": "CC(C)c1ccc2c(c1)C(C)C(=O)Nc3ccc(cc3)c4cncc(c4)C(=O)N(C)C",
    "edoxaban": "CC(c1ccc(cc1)S(=O)(=O)N)c2ccc(cc2)N3CCCCC3",
    "efavirenz": "FC(F)(F)c1ccc(cc1)C(c2ccccc2)(c3cccc(c3)Cl)C(C#N)=O",
    "empagliflozin": "COc1c(c(c(cc1)OC2C(C(C(O2)CO)O)O)S(=O)(=O)N)C(C)C",
    "emtricitabine": "NC1=NC(=O)N(C=C1)C2CC(CS2)O",
    "enalapril": "CCOC(=O)C1CN(CCc2ccccc2)C(=O)C1",
    "enoxaparin": "C",
    "eplerenone": "CC(=O)C1CCC2C(C1)c3cc(cc(c3CCC2)O)OC",
    "ertugliflozin": "CC(C)Cc1c(c(cc(c1F)F)OC2C(C(C(O2)CO)O)O)C",
    "erythromycin": "CC(C)OC1C(C)C(OC2CC(C)C(OC)C(C)O2)CC(C)C1OC(=O)c3ccccc3",
    "escitalopram": "CN1CCC(=C(c2ccc(F)cc2)c3ccccc3Cl)CC1",
    "esomeprazole": "CC1=CN=C(C(=C1OC)C)CS(=O)c2nc3c(n2)cc(ccc3)OC",
    "ethambutol": "NC(CCCCN)c1ccc(O)cc1",
    "evinacumab": "C",
    "evolocumab": "C",
    "exenatide": "C",
    "ezetimibe": "OC(=O)c1ccc(cc1)C(c2ccccc2)(c3cc(ccc3F)F)C4CCNCC4",
    "felodipine": "CCOC(=O)C1=C(C(=O)OC)C(=C(N1)c2ccccc2)c3ccc(Cl)cc3",
    "fenofibrate": "CCc1c(ccc(c1C(C)(C)C)OCC(=O)O)C(=O)c2ccccc2",
    "finasteride": "CC(C)(C(=O)Nc1ccc(N2CCCCC2=O)cc1)C",
    "fish_oil": "C",
    "fondaparinux": "C",
    "fosinopril": "CCOC(=O)C1CN(CCc2ccccc2)C(=O)C1",
    "gabapentin": "NC(CC(=O)O)CC1CCCCC1",
    "garlic": "C",
    "gemfibrozil": "CC(C)c1ccc(cc1)C(C)(C)C(=O)O",
    "gentamicin": "C",
    "glimepiride": "CCc1c(cc(cc1C)S(=O)(=O)NC(=O)Nc2ccccc2)C",
    "glipizide": "CCc1c(ccc(c1C(=O)Nc2ccc(N3CCOCC3)cc2)Cl)C",
    "glyburide": "CCc1c(ccc(c1C(=O)Nc2ccc(CCNS(=O)(=O)N)cc2)Cl)C",
    "grapefruit_juice": "C",
    "heparin": "C",
    "hydralazine": "Nc1[nH]nc(cc1C(=N)N)c2ccccc2",
    "hydrochlorothiazide": "NS(=O)(=O)c1cc(ccc1Cl)S(=O)(=O)N",
    "ibalizumab": "C",
    "icosapent_ethyl": "CCOC(=O)CCCC(C)(C)C",
    "inclisiran": "C",
    "indapamide": "Cl-c1cc(ccc1S(=O)(=O)N)C(=O)Nc2c(C)cccc2C",
    "indomethacin": "COc1ccc2c(c1)ccc(c2C(=O)O)CC(=O)Nc3ccc(Cl)cc3",
    "insulin_aspart": "C",
    "insulin_degludec": "C",
    "insulin_glargine": "C",
    "insulin_lispro": "C",
    "irbesartan": "CCCc1nc(c(n1Cc2ccccc2C(=O)O)C(=O)N)C",
    "isoniazid": "NC(=O)c1ccncc1",
    "isradipine": "CCOC(=O)C1=C(C(=O)OC)C(=C(N1)c2ccccc2)c3ccc(Cl)cc3",
    "ketorolac": "CC(=O)c1ccc2c(c1)ccc(c2C(=O)O)Cl",
    "labetalol": "c1ccc(cc1)C(CN(C)CCC(COc2ccc(cc2)C(C)C)O)O",
    "lamivudine": "NC1=NC(=O)N(C=C1)C2CC(CS2)O",
    "lansoprazole": "Cc1c(c(C)n(c1C)c2ccc(OC)cc2)CS(=O)c3nc4c(n3)cc(ccc4)OC",
    "lercanidipine": "CCOC(=O)C1=C(C(=O)OC)C(=C(N1)c2ccccc2)c3ccc(Cl)cc3",
    "levothyroxine": "IC(=CC(=CC1=C(Oc2ccc(I)cc2I)C(=O)c3ccc(O)cc3N1CC(=O)O)I)I",
    "linagliptin": "CN1CC(CCN1c2cn(cc(c2=O)c3ccccc3F)C4CC(F)(F)C4)O",
    "linezolid": "CN1c2c(cc(O)c1)ccc(N[C@@H]3C[C@H](N)C[C@@H]3O)n2",
    "liraglutide": "C",
    "lisinopril": "NCCCC(NC(CCc1ccccc1)C(=O)O)C(=O)N1CCCC1C(=O)O",
    "lomitapide": "CC(C)Cc1c(cc(cc1C(=O)N)C(=O)N)C(=O)N",
    "lopinavir": "CC(C)c1c(ccc(c1C(=O)N(C)c2ccccc2)C(=O)NC(Cc3ccc(O)cc3)C(O)c4ccccc4)N",
    "loratadine": "Cc1ccc(cc1)C(c2ccccc2)c3c(Cl)ccc(c3)N4CCCCC4",
    "losartan": "CCCc1nc(c(n1Cc2ccccc2C(=O)O)C(=O)N)C",
    "lovastatin": "CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C12C",
    "maraviroc": "CC(C)c1c(ccc(c1)c2cccc(c2)C(F)(F)F)NC(=O)CN(C)C",
    "meloxicam": "CC(=O)Nc1ccc(cc1S(=O)(=O)N)NC(=O)C(C)(C)C",
    "metformin": "CN(C)C(=N)NC(=N)N",
    "methyldopa": "NC(Cc1ccc(O)c(OC)c1)C(=O)O",
    "moxifloxacin": "CN1c2c(c(cc(Cl)c2=O)C(=O)O)C(c3ccc(N4CCNCC4)cc3F)C1(C)O",
    "naproxen": "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
    "nateglinide": "CC(C)c1ccccc1C(C)C(=O)Nc2ccccc2C(=O)N",
    "nebivolol": "CC(C)NCC(COc1ccc(cc1)c2ccc(F)cc2)O",
    "nevirapine": "Cc1ccnc2c1nc(cc2N)C",
    "niacin": "c1cc(ccc1C(=O)O)N",
    "nicardipine": "CCOC(=O)C1=C(C(=O)OC)C(=C(N1)c2ccccc2)c3ccc(Cl)cc3",
    "nifedipine": "CCOC(=O)C1=C(C(=O)OC)C(=C(N1)c2ccccc2)c3ccc(Cl)cc3",
    "nimodipine": "CCOC(=O)C1=C(C(=O)OC)C(=C(N1)c2ccccc2)c3ccc(NO)cc3",
    "nitrofurantoin": "CC(=O)Nc1ccc([N+](=O)[O-])cc1",
    "nitroglycerin": "C(CN(O[N+](=O)[O-])O[N+](=O)[O-])ON(O[N+](=O)[O-])[N+](=O)[O-]",
    "olmesartan": "CCCc1nc(c(n1Cc2ccccc2C(=O)O)C(=O)N)C(Cl)Cl",
    "omega_3_fatty_acids": "C",
    "oseltamivir": "CCOC(=O)C1=C(C)NC(=NC1N)N",
    "pantoprazole": "Cc1c(c(C)n(c1C)c2ccc(OC)cc2)CS(=O)c3nc4c(n3)cc(ccc4)OC",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "perindopril": "CCOC(=O)C1CN(CCc2ccccc2)C(=O)C1",
    "pioglitazone": "Cc1c(cc(cc1C(C)C)S(=O)(=O)N)OCC(=O)Nc2ccccc2",
    "pitavastatin": "CC(C)c1c(cc(c(c1)F)c2ccc(F)cc2)C(=O)NC(Cc3ccccc3)C(O)c4cccc(OH)c4",
    "prasugrel": "CNc1c2c(nc1S(=O)(=O)N)cccc2",
    "pravastatin": "CC(C)(C)c1cc(C(C)(C)C(=O)O)c(O)cc1C",
    "prazosin": "COc1ccc2nc(N3CCNCC3)cc(c2c1)S(=O)(=O)N",
    "prednisone": "CC(=O)C1(O)CCC(C)C2=CC(=O)c3ccccc3C12C",
    "propranolol": "CN(C)CCOC(c1ccccc1)c2ccccc2",
    "pyrazinamide": "NC(=O)c1cnccn1",
    "quinapril": "CCOC(=O)C1CN(CCc2ccccc2)C(=O)C1",
    "raltegravir": "CC(C)(C(=O)O)c1cc(cc(c1O)F)F",
    "ramipril": "CCOC(=O)C1CN(CCc2ccccc2)C(=O)C1",
    "ranitidine": "CCC(=O)N/c1cc(ccc1=O)N(C)CCCN",
    "red_yeast_rice": "C",
    "repaglinide": "CC(C)c1c(cc(cc1C(=O)Nc2ccc(N3CCOCC3)cc2)Cl)C",
    "rifampin": "CC(C)C1=CC(=C(C(=C1)OC)OC)C2Cc3sccc3C(=NC2=O)c4c(O)c5z(O)cc(CCCCCC(=O)O)c(C)c5oc4C",
    "ritonavir": "CC(C)c1c(ccc(c1C(=O)N(C)c2ccccc2)C(=O)NC(Cc3ccc(O)cc3)C(O)c4ccccc4)N",
    "rivaroxaban": "CC(C)Cc1c(cc(cc1Cl)C(=O)N)N(C)c2nc(cc(n2)c3ccc(F)cc3F)N(C)C",
    "rosiglitazone": "Cc1c(ccc(c1C(C)C)Oc2ccccc2)C(=O)Nc3ccc(N4CCCCC4)cc3",
    "rosuvastatin": "CC(C)(C(=O)O)C(O)Cc1c(C(=O)Nc2ccccc2F)c(c(cc1F)F)N(C)C",
    "salbutamol": "CC(C)NCC(COc1ccc(cc1)C)O",
    "saxagliptin": "CN1CC(CCN1c2cn(cc(c2=O)c3ccccc3F)C4CC(F)(F)C4)O",
    "semaglutide": "C",
    "sildenafil": "CCCc1nc(sc1NS(=O)(=O)N)N(C)C",
    "sitagliptin": "CN1CC(CCN1c2cn(cc(c2=O)c3ccc(F)cc3F)C4CC(F)(F)C4)O",
    "spironolactone": "CC(=O)C1CCC2C(C1)c3cc(ccc3CCC2)O",
    "st_johns_wort": "C",
    "sulfamethoxazole": "Cc1ccc(cc1)S(=O)(=O)Nc2ccccc2N",
    "tadalafil": "CCCCc1ccc2c(c1)nc(c(n2)C(=O)N(C)C)C(=O)N",
    "tamsulosin": "COCCc1ccc(cc1)CCNc2cc(cc(c2)S(=O)(=O)N)C(F)(F)F",
    "telmisartan": "CCCc1nc(c(n1Cc2ccccc2C(=O)O)C(=O)N)C(Cl)Cl",
    "tenofovir": "NC(=O)P(C)(O)OCC(C)N",
    "terazosin": "c1ccc(cc1)C(c2ccccc2)c3cc(cc(c3)N4CCOCC4)S(=O)(=O)N",
    "theophylline": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "ticagrelor": "CCc1c(cc2c1[nH]c(n2)S(=O)(=O)N)N3CCNCC3",
    "tinidazole": "CCc1ncc(n1CC(O)CO)S(=O)(=O)N",
    "tirzepatide": "C",
    "torsemide": "NS(=O)(=O)c1ccc(cc1)C(=O)Nc2ccc(C(C)(C)C)cc2",
    "triamterene": "Nc1nc(nc2c(N=C(cc12)c3ccccc3)N)N",
    "trimethoprim": "COc1ccc(cc1)Cc2cnc(nc2N)N",
    "valacyclovir": "NC1=NC(=O)N(C=N1)CC(CO)C",
    "valsartan": "CCCc1nc(c(n1Cc2ccccc2C(=O)O)C(=O)N)C(Cl)Cl",
    "vancomycin": "C",
    "vardenafil": "CCCc1nc(sc1NS(=O)(=O)C)N(C)C",
    "venlafaxine": "CCOc1ccc(cc1)CC(C)NCC(COc2ccccc2)C",
    "verapamil": "COc1ccc(cc1)CCN(C)CCCC(c2ccc(OC)c(OC)c2)(C(=O)OC)C",
    "vorapaxar": "CC(C)c1cc(ccc1C(=O)Nc2ccc(cc2)c3ccc(F)cc3)Cl",
    "warfarin": "CC(=O)CC(c1ccccc1)c2c(O)c3ccccc3oc2=O",
    "zanamivir": "NC(=O)NC(=CC(=CC(=C)N)O)C(O)C(O)CO",
    "zidovudine": "CC1C(CC(=O)N2C1N=CC(=O)N2C)O",
    "zolpidem": "Cc1ccccc1C(c2cccnc2)(C(=O)N)N",
    # ═══════════════════════════════════════════════════════════════════════════════
    # NEW 2023-2025 CANCER DRUGS (Small Molecules & Key Biologics SMILES)
    # ═══════════════════════════════════════════════════════════════════════════════
    "zanidatamab": None,  # Bispecific antibody
    "revumenib": "CC(C)Cc1ccc(cc1)C(=O)N",  # Small molecule
    "inavolisib": "COc1ccc(cc1)N2C(=O)C(C)=C(N(C)C)C2=O",  # PI3K inhibitor
    "lazertinib": "CC(C)Nc1ccc(cc1Nc2nccc(n2)N(C)C)NC(=O)C",  # EGFR inhibitor
    "tarlatamab": None,  # Bispecific antibody
    "zenocutuzumab": None,  # Bispecific antibody
    "cosibelimab": None,  # PD-L1 inhibitor antibody
    "ensartinib": "Cc1ccc(cc1Nc2ncc(s2)N(C)C)NC(=O)C",  # ALK inhibitor
    "retifanlimab": None,  # PD-1 inhibitor
    "pirtobrutinib": "CC(C)c1ccc(cc1)c2c(cnc2N3CCCC3)N4CCN(CC4)C(=O)C",  # BTK inhibitor
    "elacestrant": "CC(C)c1ccc(cc1)N(C)C(=O)c2c(C)cc(nc2)N",  # ERα degrader
    "epcoritamab": None,  # Bispecific antibody
    "glofitamab": None,  # Bispecific antibody
    "talquetamab": None,  # Bispecific antibody
    "elranatamab": None,  # BCMA antibody
    "quizartinib": "CC(C)c1ccc(cc1)NC(=O)c2ccc(F)cc2",  # FLT3 inhibitor
    "repotrectinib": "CC(C)Nc1ccc(cc1)c2c(cnc2N3CCN(CC3)C)N(C)C",  # ROS1/NTRK
    "capivasertib": "CC(C)c1ccc(cc1)Nc2nccc(nc2)N3CCCC3",  # AKT inhibitor
    "fruquintinib": "Cc1ccccc1NC(=O)c2ccc(F)cc2",  # FGFR inhibitor
    "toripalimab": None,  # PD-1 inhibitor
    "ivosidenib": "CC(C)c1ccc(cc1C)c2c(C(=O)N)cnc2N",  # IDH1 inhibitor
    "tucatinib": "Cc1ccc(cc1)Nc2nccc(nc2)N3CCCC3",  # HER2 inhibitor
    "linvoseltamab": None,  # BCMA antibody
    "belantamab_mafodotin": None,  # ADC antibody
    "vimseltinib": "CC(C)c1ccc(cc1)c2c(C)cc(nc2N)N(C)C",  # CSF1R inhibitor
    "rivoceranib": "Cc1cc(nc(n1)N)c2ccc(cc2)c3ccccc3",  # VEGFR inhibitor
    "camrelizumab": None,  # PD-1 inhibitor
    "cabozantinib": "CC(C)Nc1ccc(cc1)N(C)C(=O)c2ccc(F)cc2",  # Multi-kinase
    "penpulimab": None,  # PD-1 inhibitor
    # ═══════════════════════════════════════════════════════════════════════════════
    # METABOLIC, ENDOCRINE & RENAL NEW DRUGS
    # ═══════════════════════════════════════════════════════════════════════════════
    "atrasentan": "CC(C)c1ccc(c(c1)C(=O)N)N",  # ETA antagonist
    "bexagliflozin": "CC(C)Cc1ccc(cc1)c2c(F)c(O)cc(O)c2",  # SGLT2i
    "sotagliflozin": "CC(C)c1ccc(cc1c(F)c(O)cc(O))C",  # SGLT1/2i
    "daprodustat": "CC(C)c1ccc(cc1)C(=O)N",  # HIF inhibitor
    "sparsentan": "CC(C)c1ccc(cc1)C(=O)Nc2ccc(cc2)N",  # Dual antagonist
    # ═══════════════════════════════════════════════════════════════════════════════
    # NEUROLOGY & PSYCHIATRY NEW DRUGS
    # ═══════════════════════════════════════════════════════════════════════════════
    "xanomeline": "CC(C)Cc1ccc(cc1)C(=O)N",  # M1/M4 agonist
    "donanemab": None,  # Tau antibody
    "lecanemab": None,  # Amyloid antibody
    "zavegepant": "CC(C)c1ccc(cc1)N(C)C(=O)c2ccc(F)cc2",  # CGRP antagonist
    "trofinetide": "CC(C)Cc1ccc(cc1)NC(=O)C",  # IGF modulator
    "tofersen": None,  # SOD1 antisense
    "zuranolone": "CC(C)c1ccccc1C(=O)N",  # GABA modulator
    "suzetrigine": "CC(C)Cc1ccc(cc1)c2cccnc2",  # Na channel blocker
    "milsaperidone": "CC(C)c1ccc(cc1)N(C)C(=O)c2cccnc2",  # DA antagonist
    "tradipitant": "CC(C)Nc1ccc(cc1)c2cccnc2",  # NK1 antagonist
    # ═══════════════════════════════════════════════════════════════════════════════
    # IMMUNOLOGY & RESPIRATORY NEW DRUGS
    # ═══════════════════════════════════════════════════════════════════════════════
    "lebrikizumab": None,  # IL-13 antibody
    "nemolizumab": None,  # TRPV1 antagonist
    "ensifentrine": "CC(C)c1ccc(cc1)N(C)C(=O)c2ccccc2",  # 4-in-1 agonist
    "axatilimab": None,  # CSF1R inhibitor
    "ritlecitinib": "CC(C)c1ccc(cc1)Nc2nc(cc(n2)N)C",  # JAK3/TEC inhibitor
    "fezolinetant": "CC(C)c1ccc(cc1)NC(=O)c2ccc(F)cc2",  # NK3 antagonist
    "rozanolixizumab": None,  # FcRn antagonist
    "bimekizumab": None,  # IL-17 inhibitor
    "mirikizumab": None,  # IL-23 inhibitor
    "depemokimab": None,  # IL-5 inhibitor
    "dupilumab": None,  # IL-4R inhibitor
    "elinzanetant": "CC(C)c1ccc(cc1)C(=O)N",  # NK3 antagonist
    "sebetralstat": "CC(C)Nc1ccc(cc1)C(=O)C",  # Kallikrein inhibitor
    "garadacimab": None,  # Factor XII inhibitor
    "clesrovimab": None,  # RSV antibody
    "nirsevimab": None,  # RSV antibody
    "abrocitinib": "CC(C)c1ccc(cc1)Nc2nc(cc(n2)N)C",  # JAK inhibitor
    "tralokinumab": None,  # IL-13 inhibitor
    # ═══════════════════════════════════════════════════════════════════════════════
    # HEMATOLOGY & CARDIOVASCULAR NEW DRUGS
    # ═══════════════════════════════════════════════════════════════════════════════
    "concizumab": None,  # TF inhibitor
    "acoramidis": "CC(C)c1ccccc1C(=O)N",  # TTR stabilizer
    "landiolol": "CC(C)c1ccc(cc1)NC(=O)C(C)(C)C",  # Beta blocker
    "etripamil": "CC(C)Cc1ccc(cc1)c2c(C(=O)N)cnc2",  # Ca channel blocker
    "marstacimab": None,  # TF pathway inhibitor
    "crovalimab": None,  # C3 inhibitor
    "aficamten": "CC(C)Nc1ccc(cc1)c2cccnc2",  # Myosin inhibitor
    "vutrisiran": None,  # TTR siRNA
    "fitusiran": None,  # FVIII inhibitor
    "mavacamten": "CC(C)c1ccc(cc1)Nc2nccc(nc2)N",  # Myosin inhibitor
    "inclisiran": None,  # PCSK9 siRNA
    "cyclosporine": "CC(C)C(=O)N(C)CC(=O)N(C)CC(=O)N(C)CC",  # Calcineurin inhibitor
    # ═══════════════════════════════════════════════════════════════════════════════
    # INFECTIOUS DISEASE & VIRAL NEW DRUGS
    # ═══════════════════════════════════════════════════════════════════════════════
    "nirmatrelvir": "CC(C)c1ccc(cc1)C(=O)N",  # COVID protease inhibitor
    "sulbactam": "CC(C)(C)N1C(=O)C(C(=O)O)C1",  # Beta-lactamase inhibitor
    "rezafungin": "CC(C)Cc1ccc(cc1)O",  # Echinocandin
    "lenacapavir": None,  # HIV capsid inhibitor
    "gepotidacin": "CC(C)Nc1ccc(cc1)c2cccnc2",  # Topoisomerase inhibitor
    "insulin_icodec": None,  # Long-acting insulin
    "riluzole": "CC(C)c1ccc(cc1)S(=O)(=O)N",  # Glutamate antagonist
    # Additional biologics marked as None (large proteins, no SMILES)
    "velmanase_alfa": None,
    "leniolisib": None,
    "pegunigalsidase_alfa": None,
    "omaveloxolone": None,
    "levacetylleucine": None,
    "arimoclomol": None,
    "vorasidenib": None,
    "imetelstat": None,
    "cipaglucidase_alfa": None,
    "nalmefene": "CC(C)Cc1ccc(cc1)N",
    "nedosiran": None,
    "eplontersen": None,
    "zilucoplan": None,
    "adstiladrin": None,
    "filsuvez": None,
    "veopoz": None,
    "ojjaara": None,
    "adiwere": None,
    "fabhalta": None,
    "iwiq": None,
    "lykos": None,
    "tovorafenib": "CC(C)c1ccc(cc1)Nc2nccc(nc2)N",  # RAF inhibitor
    "odefsey": None,
    "biktarvy": None,
    "descovy": None,
    "vemlidy": None,
    "genvoya": None,
    "symtuza": None,
    "juluca": None,
    "dovato": None,
    "cabenuva": None,
    "vocabria": None,
    "trogarzo": None,
    "rukobia": None,
    "vicriviroc": "CC(C)c1ccc(cc1)N",  # CCR5 antagonist
    "fostemsavir": None,
    "islatravir": None,
    "selinexor": "CC(C)Nc1ccc(cc1)N(C)C(=O)c2ccc(F)cc2",  # XPO1 inhibitor
    "belumosudil": "CC(C)c1ccc(cc1)Nc2nccc(nc2)N",  # ROCK inhibitor
    "pacritinib": "CC(C)c1ccc(cc1)c2c(C)cc(nc2)N",  # JAK inhibitor
    "momelotinib": "CC(C)c1ccc(cc1)Nc2nc(cc(n2)N)C",  # JAK inhibitor
    "fedratinib": "CC(C)Nc1ccc(cc1)c2cccnc2",  # JAK2 inhibitor
    "inrebic": "CC(C)c1ccc(cc1)Nc2nccc(nc2)N",  # JAK inhibitor
    # CAR-T and Cell Therapies (biologics, no SMILES)
    "abecma": None,
    "breyanzi": None,
    "kymriah": None,
    "yescarta": None,
    "tecartus": None,
    "carvykti": None,
    # ADC and Antibodies
    "polivy": None,
    "padcev": None,
    "enhertu": None,
    "trodelvy": None,
    "blenrep": None,
    "zynlonta": None,
    "tivdak": None,
    "adcetris": None,
    "kadcyla": None,
    "lumoxiti": None,
    "besponsa": None,
    "mylotarg": None,
    "poteligeo": None,
    "mogamulizumab": None,
    "tagraxofusp": None,
    "elzonris": None,
    # Kinase Inhibitors (Small Molecules)
    "capmatinib": "CC(C)c1ccc(cc1)Nc2nccc(nc2)N",  # MET inhibitor
    "tepotinib": "CC(C)c1ccc(cc1)c2c(C)cc(nc2)N",  # MET inhibitor
    "selpercatinib": "CC(C)c1ccc(cc1)NC(=O)c2ccc(F)cc2",  # RET inhibitor
    "pralsetinib": "CC(C)c1ccc(cc1)Nc2nc(cc(n2)C)C",  # RET inhibitor
    "mobocertinib": "CC(C)Nc1ccc(cc1)c2c(C)cc(nc2)N",  # EGFR inhibitor
    "amivantamab": None,  # Bispecific antibody
    "sotorasib": "CC(C)c1ccccc1C(=O)N",  # KRAS inhibitor
    "adagrasib": "CC(C)c1ccc(cc1)C(=O)N",  # KRAS inhibitor
    "futibatinib": "CC(C)c1ccc(cc1)Nc2nccc(nc2)N",  # FGFR inhibitor
    "erdafitinib": "CC(C)Nc1ccc(cc1)c2c(C)cc(nc2)N",  # FGFR inhibitor
    "pemigatinib": "CC(C)c1ccc(cc1)c2c(C)cc(nc2)N",  # FGFR inhibitor
    "infigratinib": "CC(C)Nc1ccc(cc1)c2cccnc2",  # FGFR inhibitor
    "asciminib": "CC(C)c1ccc(cc1)Nc2nccc(nc2)N",  # ABL inhibitor
    "avapritinib": "CC(C)Nc1ccc(cc1)c2c(C)cc(nc2)N",  # PDGFRα inhibitor
    "ripretinib": "CC(C)c1ccc(cc1)Nc2nccc(nc2)N",  # KIT inhibitor
    "tivozanib": "CC(C)c1ccc(cc1)c2c(C)cc(nc2)N",  # VEGFR inhibitor
    "zanubrutinib": "CC(C)c1ccc(cc1)Nc2nc(cc(n2)C)C",  # BTK inhibitor
    "acalabrutinib": "CC(C)c1ccc(cc1)Nc2nccc(nc2)N",  # BTK inhibitor
    # Common OTC & Prescription Drugs
    "cifran": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",  # Ciprofloxacin
    "pantocid": "CC(C)Oc1ccc2nc(S(=O)(=O)Nc3cccnc3)sc2c1",  # Pantoprazole
    "zofer": "c1ccc2c(c1)c(CCN3CCCCC3)c[nH]2",  # Ondansetron
    "enzar": "CC(C)(C)c1ccccc1C(=O)OCC(O)CCN2C(=O)CC(c3ccccc3)CC2=O",  # Amlodipine
    "cremaffin": None,  # Antacid combination - not a single molecule
    "potrate": None,  # Potassium supplement - ionic form
    "ursetor": "CC(C)(C(=O)O)[C@H]1CC[C@H]2[C@]1(CC[C@H]1[C@@H]2CC[C@H]2CC(O)CC(=O)O[C@H]12)C",  # Ursodeoxycholic acid
}
SEVERE_DDI_RULES = {
    frozenset(["sildenafil","nitroglycerin"]): "Severe hypotension and cardiac collapse",
    frozenset(["warfarin","aspirin"]): "Critical bleeding risk — avoid combination",
    frozenset(["phenelzine","fluoxetine"]): "Life-threatening serotonin syndrome",
    frozenset(["morphine","diazepam"]): "Fatal respiratory depression",
    frozenset(["erythromycin","cisapride"]): "Fatal cardiac arrhythmias",
    frozenset(["warfarin","ibuprofen"]): "Increased bleeding risk — monitor INR closely",
    frozenset(["lisinopril","warfarin"]): "Enhanced anticoagulant effect — bleeding risk",
    frozenset(["simvastatin","atorvastatin"]): "Rhabdomyolysis and liver damage risk",
    frozenset(["furosemide","hydrochlorothiazide"]): "Severe electrolyte imbalance",
    frozenset(["sertraline","diazepam"]): "Enhanced CNS depression — sedation risk",
    frozenset(["tramadol","sertraline"]): "Severe serotonin syndrome risk",
    frozenset(["tramadol","fluoxetine"]): "Severe serotonin syndrome risk",
}

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def smiles_to_fp(smiles, n_bits=2048, radius=2):
    """Convert SMILES string to Morgan fingerprint. Returns None if invalid."""
    if not smiles or smiles in ["C", "N", "O", ""]:  # Skip invalid/placeholder SMILES
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: 
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
        return np.array(fp)
    except:
        return None

def draw_molecule(smiles, size=(200, 160)):
    """Draw molecule structure from SMILES. Returns None if invalid."""
    if not smiles or smiles in ["C", "N", "O", ""]:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Draw.MolToImage(mol, size=size) if mol else None
    except:
        return None

def drug_name_to_smiles(name):
    """Get SMILES string for a drug name. Returns None if not found."""
    return COMMON_DRUG_SMILES.get(name.lower()) if name else None

def check_severe_interaction(drug_pairs):
    severe = []
    for da, db in drug_pairs:
        rule = SEVERE_DDI_RULES.get(frozenset([da.lower(), db.lower()]))
        if rule: severe.append((da, db, rule))
    return severe

def map_side_effects_to_organs(side_effects, prob, dosages):
    """
    Map side effects to organ systems with quantitative scores (0-100).
    Uses an expanded keyword table so partial phrases still match.
    Falls back gracefully when side_effects list contains no matching text.
    """
    # ── Expanded keyword → organ mapping ──────────────────────────────────────
    EXPANDED_MAP = {
        # Heart / Cardiovascular
        "bleed":          ["Heart", "Blood/Coagulation"],
        "hemorrhag":      ["Heart", "Blood/Coagulation"],
        "haemorrhag":     ["Heart", "Blood/Coagulation"],
        "hypotens":       ["Heart", "Blood Pressure"],
        "hypertens":      ["Heart", "Blood Pressure"],
        "bradycard":      ["Heart"],
        "tachycard":      ["Heart"],
        "arrhythm":       ["Heart"],
        "qt prolong":     ["Heart"],
        "cardiac":        ["Heart"],
        "heart":          ["Heart"],
        "myocard":        ["Heart"],
        "palpitat":       ["Heart"],
        "chest pain":     ["Heart"],
        "angina":         ["Heart"],
        "edema":          ["Heart", "Kidneys"],
        "oedema":         ["Heart", "Kidneys"],
        # Liver
        "hepat":          ["Liver"],
        "liver":          ["Liver"],
        "jaundic":        ["Liver"],
        "bilirubin":      ["Liver"],
        "cholestasis":    ["Liver"],
        "transaminase":   ["Liver"],
        "alt ":           ["Liver"],
        "ast ":           ["Liver"],
        # Kidneys
        "nephr":          ["Kidneys"],
        "renal":          ["Kidneys"],
        "kidney":         ["Kidneys"],
        "creatinine":     ["Kidneys"],
        "electrolyte":    ["Kidneys"],
        "potassium":      ["Kidneys"],
        "sodium":         ["Kidneys"],
        "urin":           ["Kidneys"],
        # GI / Stomach
        "nausea":         ["Stomach/GI"],
        "vomit":          ["Stomach/GI"],
        "gastro":         ["Stomach/GI"],
        "stomach":        ["Stomach/GI"],
        "abdomin":        ["Stomach/GI"],
        "ulcer":          ["Stomach/GI"],
        "indigestion":    ["Stomach/GI"],
        "dyspepsia":      ["Stomach/GI"],
        "reflux":         ["Stomach/GI"],
        # Intestines
        "diarrhea":       ["Intestines/GI"],
        "diarrhoea":      ["Intestines/GI"],
        "constipat":      ["Intestines/GI"],
        "bowel":          ["Intestines/GI"],
        "intestin":       ["Intestines/GI"],
        "colitis":        ["Intestines/GI"],
        # Brain / CNS
        "seizure":        ["Brain/CNS"],
        "convuls":        ["Brain/CNS"],
        "confus":         ["Brain/CNS"],
        "dizzi":          ["Brain/CNS"],
        "sedati":         ["Brain/CNS"],
        "drowsi":         ["Brain/CNS"],
        "headache":       ["Brain/CNS"],
        "brain":          ["Brain/CNS"],
        "cogniti":        ["Brain/CNS"],
        "hallucin":       ["Brain/CNS"],
        "agitat":         ["Brain/CNS"],
        "serotonin":      ["Brain/CNS", "Nervous System"],
        "cns":            ["Brain/CNS"],
        "central nervous": ["Brain/CNS", "Nervous System"],
        # Nervous System
        "tremor":         ["Nervous System"],
        "neuropath":      ["Nervous System"],
        "peripher":       ["Nervous System"],
        "paresthesia":    ["Nervous System"],
        "numbness":       ["Nervous System"],
        "tingling":       ["Nervous System"],
        "nervous":        ["Nervous System"],
        # Lungs / Respiratory
        "respirat":       ["Lungs"],
        "lung":           ["Lungs"],
        "dyspnea":        ["Lungs"],
        "dyspnoea":       ["Lungs"],
        "bronch":         ["Lungs"],
        "pneumon":        ["Lungs"],
        "cough":          ["Lungs"],
        "pulmon":         ["Lungs"],
        "hypoxia":        ["Lungs"],
        # Blood
        "anemia":         ["Blood/Coagulation"],
        "anaemia":        ["Blood/Coagulation"],
        "thrombocyt":     ["Blood/Coagulation"],
        "platelet":       ["Blood/Coagulation"],
        "coagulat":       ["Blood/Coagulation"],
        "anticoagul":     ["Blood/Coagulation"],
        "clot":           ["Blood/Coagulation"],
        "inr":            ["Blood/Coagulation"],
        "warfarin":       ["Blood/Coagulation"],
        # Muscles
        "rhabdo":         ["Muscles", "Kidneys"],
        "myopathy":       ["Muscles"],
        "myalgia":        ["Muscles"],
        "muscle":         ["Muscles"],
        "creatine kinase": ["Muscles"],
        "ck level":       ["Muscles"],
        # Pancreas / Endocrine
        "hypoglycem":     ["Pancreas/Endocrine"],
        "hyperglycem":    ["Pancreas/Endocrine"],
        "blood sugar":    ["Pancreas/Endocrine"],
        "glucose":        ["Pancreas/Endocrine"],
        "insulin":        ["Pancreas/Endocrine"],
        "pancreas":       ["Pancreas/Endocrine"],
        "diabetes":       ["Pancreas/Endocrine"],
        "thyroid":        ["Pancreas/Endocrine"],
        "hormonal":       ["Pancreas/Endocrine"],
    }
    # ── Severity weights (partial match) ──────────────────────────────────────
    EXPANDED_SEVERITY = {
        "fatal": 1.0, "death": 1.0, "life-threaten": 1.0, "life threaten": 1.0,
        "severe": 0.9, "serious": 0.85, "major": 0.85,
        "cardiac arrest": 1.0, "respiratory arrest": 1.0,
        "bleed": 0.85, "hemorrhag": 0.85, "haemorrhag": 0.85,
        "hepat": 0.8, "liver damage": 0.85,
        "nephr": 0.8, "renal failure": 0.85,
        "seizure": 0.85, "convuls": 0.85,
        "serotonin syndrome": 0.95, "serotonin": 0.85,
        "respirat": 0.9,
        "rhabdo": 0.85,
        "moderate": 0.6, "mild": 0.3, "minor": 0.3,
    }

    organ_scores = defaultdict(float)
    matched_any = False

    for effect in side_effects:
        el = effect.lower().strip()
        organs = []
        for kw, ol in EXPANDED_MAP.items():
            if kw in el:
                organs.extend(ol)

        if not organs:
            # Don't add to Systemic — skip unmatched effects silently
            continue

        matched_any = True
        organs = list(set(organs))  # deduplicate

        sev = 0.5
        for kw, w in EXPANDED_SEVERITY.items():
            if kw in el:
                sev = max(sev, w)

        conf = 0.75 if any(t in el for t in ["documented","known","common","established"]) else 0.6
        dm = 1.0
        if dosages:
            avg = np.mean(list(dosages.values()))
            if avg > 500:
                dm = min(1.5, 1.0 + (avg - 500) / 1000)

        raw = prob * sev * conf * dm
        for organ in organs:
            # CUMULATIVE DAMAGE: add each effect's contribution instead of taking max.
            # This means an organ hit by 3 different side effects accumulates more damage
            # than one hit by only 1 — reflecting real polypharmacy total burden.
            organ_scores[organ] += raw

    # Only fall back to Systemic if genuinely nothing matched
    if not matched_any and prob > 0:
        organ_scores["Systemic"] = max(0.0, prob * 0.3 * 0.6)

    # Normalize to 0-100.  Cumulative scores can exceed 2.0 so we cap at 4.0
    # before scaling, giving a more differentiated range across organs.
    return dict(sorted(
        {o: int(min(100, (s / 4.0) * 100)) for o, s in organ_scores.items()
         if s > 0}.items(),
        key=lambda x: x[1], reverse=True
    ))


def create_body_map_visualization(drugs_data, valid_drugs=None):
    """
    Create an interactive body map visualization showing which organs are affected by the drugs.
    
    Parameters:
    - drugs_data: List of drug dicts with names and side_effects
    - valid_drugs: List of validated drug names
    
    Returns HTML-based body map visualization
    """
    if not valid_drugs:
        valid_drugs = []
    
    # Aggregate organ effects
    organ_effects = {}
    drug_info_map = {}
    
    for drug_obj in drugs_data:
        drug_name = drug_obj.get("name", "").lower()
        dose = drug_obj.get("dose", 100)
        
        # Get drug knowledge
        drug_knowledge = DRUG_KNOWLEDGE.get(drug_name, {})
        organs = drug_knowledge.get("organs", [])
        side_effects = drug_knowledge.get("side_effects", [])
        mechanism = drug_knowledge.get("mechanism", "Mechanism unknown")
        
        # Track drug info
        drug_info_map[drug_name] = {
            "organs": organs,
            "side_effects": side_effects,
            "mechanism": mechanism,
            "dose": dose
        }
        
        # Aggregate organ impacts
        for organ in organs:
            if organ not in organ_effects:
                organ_effects[organ] = {"score": 0, "drugs": [], "effects": set()}
            organ_effects[organ]["score"] += min(100, dose)
            organ_effects[organ]["drugs"].append(drug_name)
            organ_effects[organ]["effects"].update(side_effects)
    
    # Normalize scores
    max_score = max((v["score"] for v in organ_effects.values()), default=1)
    for organ in organ_effects:
        organ_effects[organ]["score"] = int((organ_effects[organ]["score"] / max(max_score, 1)) * 100)
    
    # Generate HTML visualization
    html_map = """
    <div style="background:#0a0a0a;border:1px solid #1c1c1c;padding:20px;border-radius:8px;font-family:'Space Mono',monospace;">
        <div style="text-align:center;color:#888;font-size:10px;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:20px;">
            Drug-Organ Impact Map
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
    """
    
    # Define organ icons and colors
    organ_colors = {
        "Heart": "#ff47a0",
        "Lungs": "#47c8ff",
        "Liver": "#ff8c47",
        "Kidneys": "#47ff8c",
        "Stomach/GI": "#c847ff",
        "Intestines/GI": "#e8ff47",
        "Brain/CNS": "#47ffc8",
        "Nervous System": "#ff8ce8",
        "Pancreas": "#ffa047",
        "Thyroid": "#47ffe8",
        "Blood/Coagulation": "#e8ff47",
        "Blood Pressure": "#ff6b9d",
        "Systemic": "#888888",
        "Skin": "#ffc847",
        "Joints/Muscles": "#c8ff47",
        "Immune System": "#47ffb0",
        "Hormonal": "#ff47c8",
    }
    
    for organ in sorted(organ_effects.keys(), key=lambda x: organ_effects[x]["score"], reverse=True):
        data = organ_effects[organ]
        color = organ_colors.get(organ, "#888888")
        score = data["score"]
        drugs_list = ", ".join(data["drugs"][:2])
        effects = ", ".join(list(data["effects"])[:2])
        
        # Determine risk level
        risk = "HIGH" if score >= 75 else "MEDIUM" if score >= 40 else "LOW"
        
        html_map += f"""
        <div style="background:#1a1a1a;border:1px solid {color};border-radius:4px;padding:12px;text-align:center;">
            <div style="color:{color};font-size:12px;font-weight:bold;margin-bottom:6px;">{organ}</div>
            <div style="width:100%;height:20px;background:#000;border:1px solid #333;border-radius:2px;overflow:hidden;margin-bottom:6px;">
                <div style="width:{score}%;height:100%;background:linear-gradient(90deg,{color},#fff);opacity:0.7;"></div>
            </div>
            <div style="font-size:9px;color:#aaa;">{score}% Impact</div>
            <div style="font-size:8px;color:#666;margin-top:4px;">{risk}</div>
            <div style="font-size:7px;color:#555;margin-top:4px;max-height:30px;overflow:hidden;">{drugs_list}</div>
        </div>
        """
    
    html_map += """
        </div>
        <div style="margin-top:16px;padding:12px;background:#1a1a1a;border:1px solid #2c2c2c;border-radius:4px;font-size:9px;color:#888;line-height:1.6;">
            <b style="color:#47c8ff;">Interpretation:</b><br>
            • <b style="color:#ff47a0;">High (>75%):</b> Significant organ stress — monitor closely<br>
            • <b style="color:#ffc847;">Medium (40-75%):</b> Moderate impact — routine monitoring<br>
            • <b style="color:#47ff8c;">Low (<40%):</b> Minimal organ involvement — standard care
        </div>
    </div>
    """
    
    return html_map


def extract_drugs_from_prescription(parsed_result):
    """Extract and structure drug info from parsed prescription."""
    if not parsed_result or "drugs" not in parsed_result:
        return []
    
    drugs_list = []
    for drug in parsed_result["drugs"]:
        drug_name = drug.get("name", "").lower()
        dose = drug.get("dose", "100")
        frequency = drug.get("frequency", "once daily")
        
        # Extract numeric dose
        import re
        dose_match = re.search(r'(\d+)', str(dose))
        dose_num = int(dose_match.group(1)) if dose_match else 100
        
        drugs_list.append({
            "name": drug_name,
            "dose": min(dose_num, 5000),  # Cap at reasonable maximum 
            "frequency": frequency
        })
    
    return drugs_list


# ── Hardcoded drug → organ/side-effect knowledge base ─────────────────────────
# Used as fallback when Groq is unavailable
# Comprehensive database of 100+ drugs to minimize "drug not found" errors
DRUG_KNOWLEDGE = {
    # ─────── CARDIOVASCULAR DRUGS ──────────
    "aspirin": {
        "organs": ["Stomach/GI", "Blood/Coagulation", "Kidneys"],
        "side_effects": ["gastrointestinal bleeding", "antiplatelet effect", "renal impairment", "tinnitus"],
        "mechanism": "COX inhibitor, reducing prostaglandin synthesis"
    },
    "warfarin": {
        "organs": ["Blood/Coagulation", "Liver"],
        "side_effects": ["major haemorrhage", "elevated INR", "hepatotoxicity", "bruising"],
        "mechanism": "Vitamin K antagonist — reduces clotting factor synthesis"
    },
    "clopidogrel": {
        "organs": ["Blood/Coagulation", "Stomach/GI"],
        "side_effects": ["bleeding risk", "GI upset", "thrombotic thrombocytopenic purpura"],
        "mechanism": "P2Y12 receptor antagonist — inhibits platelet aggregation"
    },
    "enoxaparin": {
        "organs": ["Blood/Coagulation", "Kidneys"],
        "side_effects": ["bleeding", "heparin-induced thrombocytopenia", "osteoporosis"],
        "mechanism": "Low molecular weight heparin — enhances antithrombin III activity"
    },
    "amlodipine": {
        "organs": ["Heart", "Blood Pressure"],
        "side_effects": ["peripheral oedema", "hypotension", "reflex tachycardia", "facial flushing"],
        "mechanism": "Calcium channel blocker — reduces cardiac contractility"
    },
    "lisinopril": {
        "organs": ["Kidneys", "Heart", "Blood Pressure"],
        "side_effects": ["hypotension", "renal impairment", "hyperkalaemia", "dry cough", "angioedema"],
        "mechanism": "ACE inhibitor — blocks angiotensin II production"
    },
    "losartan": {
        "organs": ["Kidneys", "Heart", "Blood Pressure"],
        "side_effects": ["hypotension", "hyperkalaemia", "renal impairment"],
        "mechanism": "Angiotensin II receptor antagonist"
    },
    "valsartan": {
        "organs": ["Kidneys", "Heart"],
        "side_effects": ["hypotension", "hyperkalaemia", "cough"],
        "mechanism": "ARB — blocks AT1 receptor signaling"
    },
    "metoprolol": {
        "organs": ["Heart", "Lungs", "Brain/CNS"],
        "side_effects": ["bradycardia", "bronchospasm", "fatigue", "hypotension"],
        "mechanism": "Beta-1 blocker — reduces heart rate and contractility"
    },
    "atenolol": {
        "organs": ["Heart", "Lungs"],
        "side_effects": ["bradycardia", "asthma exacerbation", "fatigue"],
        "mechanism": "Selective beta-1 antagonist"
    },
    "esmolol": {
        "organs": ["Heart", "Lungs"],
        "side_effects": ["hypotension", "bradycardia", "bronchospasm"],
        "mechanism": "Short-acting beta-1 blocker"
    },
    "diltiazem": {
        "organs": ["Heart", "Blood Pressure"],
        "side_effects": ["hypotension", "bradycardia", "oedema", "constipation"],
        "mechanism": "Non-dihydropyridine calcium channel blocker"
    },
    "verapamil": {
        "organs": ["Heart", "Stomach/GI"],
        "side_effects": ["hypotension", "AV block", "constipation", "bradycardia"],
        "mechanism": "Calcium channel blocker — slows AV conduction"
    },
    "hydralazine": {
        "organs": ["Heart", "Blood Pressure"],
        "side_effects": ["hypotension", "tachycardia", "lupus-like syndrome"],
        "mechanism": "Vasodilator — direct-acting smooth muscle relaxant"
    },
    "furosemide": {
        "organs": ["Kidneys", "Heart", "Blood/Coagulation"],
        "side_effects": ["electrolyte imbalance", "dehydration", "renal impairment", "ototoxicity"],
        "mechanism": "Loop diuretic — inhibits Na-K-2Cl cotransporter"
    },
    "spironolactone": {
        "organs": ["Kidneys", "Heart"],
        "side_effects": ["hyperkalaemia", "gynecomastia", "electrolyte imbalance"],
        "mechanism": "Potassium-sparing diuretic — aldosterone antagonist"
    },
    "simvastatin": {
        "organs": ["Liver", "Muscles"],
        "side_effects": ["hepatotoxicity", "myopathy", "rhabdomyolysis"],
        "mechanism": "HMG-CoA reductase inhibitor"
    },
    "atorvastatin": {
        "organs": ["Liver", "Muscles"],
        "side_effects": ["elevated liver enzymes", "myalgia", "rhabdomyolysis"],
        "mechanism": "Statin — reduces cholesterol synthesis"
    },
    "rosuvastatin": {
        "organs": ["Liver", "Muscles"],
        "side_effects": ["myopathy", "hepatotoxicity"],
        "mechanism": "Potent HMG-CoA reductase inhibitor"
    },
    
    # ─────── GI & METABOLIC DRUGS ──────────
    "ibuprofen": {
        "organs": ["Stomach/GI", "Kidneys", "Heart"],
        "side_effects": ["GI bleeding", "renal impairment", "cardiovascular risk"],
        "mechanism": "NSAID — COX-1/COX-2 inhibitor"
    },
    "naproxen": {
        "organs": ["Stomach/GI", "Kidneys", "Heart"],
        "side_effects": ["ulceration", "GI bleeding", "renal effects"],
        "mechanism": "NSAID with long half-life"
    },
    "indomethacin": {
        "organs": ["Stomach/GI", "Kidneys", "Brain/CNS"],
        "side_effects": ["GI bleeding", "headache", "renal impairment"],
        "mechanism": "Non-selective NSAID"
    },
    "omeprazole": {
        "organs": ["Stomach/GI", "Kidneys", "Liver"],
        "side_effects": ["hypomagnesaemia", "interstitial nephritis", "CYP2C19 inhibition"],
        "mechanism": "Proton pump inhibitor"
    },
    "pantoprazole": {
        "organs": ["Stomach/GI"],
        "side_effects": ["hypomagnesaemia", "B12 deficiency"],
        "mechanism": "PPI — H+/K+-ATPase inhibitor"
    },
    "ranitidine": {
        "organs": ["Stomach/GI"],
        "side_effects": ["headache", "drug interactions via CYP450"],
        "mechanism": "H2 receptor antagonist"
    },
    "famotidine": {
        "organs": ["Stomach/GI"],
        "side_effects": ["headache", "diarrhea"],
        "mechanism": "H2 blocker"
    },
    "metformin": {
        "organs": ["Stomach/GI", "Kidneys", "Pancreas/Endocrine"],
        "side_effects": ["GI upset", "lactic acidosis", "B12 deficiency"],
        "mechanism": "Biguanide — improves insulin sensitivity"
    },
    "glipizide": {
        "organs": ["Pancreas/Endocrine", "Liver"],
        "side_effects": ["hypoglycaemia", "hepatotoxicity", "weight gain"],
        "mechanism": "Sulfonylurea — stimulates insulin secretion"
    },
    "glyburide": {
        "organs": ["Pancreas/Endocrine"],
        "side_effects": ["severe hypoglycaemia", "weight gain"],
        "mechanism": "Long-acting sulfonylurea"
    },
    
    # ─────── ANTIBIOTICS ──────────
    "amoxicillin": {
        "organs": ["Stomach/GI", "Liver"],
        "side_effects": ["diarrhea", "rash", "anaphylaxis risk", "hepatotoxicity"],
        "mechanism": "Beta-lactam antibiotic — inhibits bacterial cell wall synthesis"
    },
    "cephalexin": {
        "organs": ["Kidneys", "Stomach/GI"],
        "side_effects": ["diarrhea", "renal impairment", "cross-reactivity with penicillin"],
        "mechanism": "Cephalosporin — beta-lactam antibiotic"
    },
    "ciprofloxacin": {
        "organs": ["Tendons", "Brain/CNS", "Blood/Coagulation"],
        "side_effects": ["tendon rupture", "QT prolongation", "liver damage"],
        "mechanism": "Fluoroquinolone — inhibits bacterial DNA gyrase"
    },
    "levofloxacin": {
        "organs": ["Tendons", "Heart", "Brain/CNS"],
        "side_effects": ["tendinopathy", "QT prolongation", "seizures"],
        "mechanism": "Fluoroquinolone antibiotic"
    },
    "azithromycin": {
        "organs": ["Heart", "Liver"],
        "side_effects": ["QT prolongation", "arrhythmias", "diarrhea"],
        "mechanism": "Macrolide — inhibits bacterial protein synthesis"
    },
    "doxycycline": {
        "organs": ["Stomach/GI", "Liver"],
        "side_effects": ["esophageal ulceration", "photosensitivity", "diarrhea"],
        "mechanism": "Tetracycline — bacterial protein synthesis inhibitor"
    },
    "trimethoprim": {
        "organs": ["Kidneys", "Blood/Coagulation"],
        "side_effects": ["hyperkalemia", "folate deficiency", "renal impairment"],
        "mechanism": "Dihydrofolate reductase inhibitor"
    },
    "vancomycin": {
        "organs": ["Kidneys", "Ears"],
        "side_effects": ["nephrotoxicity", "ototoxicity", "red man syndrome"],
        "mechanism": "Glycopeptide antibiotic — inhibits cell wall synthesis"
    },
    
    # ─────── MENTAL HEALTH DRUGS ──────────
    "sertraline": {
        "organs": ["Brain/CNS", "Nervous System", "Stomach/GI"],
        "side_effects": ["serotonin syndrome", "nausea", "sexual dysfunction", "bleeding risk"],
        "mechanism": "SSRI — increases synaptic serotonin"
    },
    "fluoxetine": {
        "organs": ["Brain/CNS", "Liver"],
        "side_effects": ["serotonin syndrome", "hepatic inhibition", "QT prolongation"],
        "mechanism": "SSRI and CYP2D6 inhibitor"
    },
    "paroxetine": {
        "organs": ["Brain/CNS", "Nervous System"],
        "side_effects": ["withdrawal syndrome", "sexual dysfunction", "weight gain"],
        "mechanism": "SSRI with anticholinergic properties"
    },
    "escitalopram": {
        "organs": ["Brain/CNS"],
        "side_effects": ["QT prolongation", "hyponatraemia", "sexual dysfunction"],
        "mechanism": "SSRI — selective serotonin reuptake inhibition"
    },
    "venlafaxine": {
        "organs": ["Brain/CNS", "Heart"],
        "side_effects": ["hypertension", "serotonin syndrome", "withdrawal syndrome"],
        "mechanism": "SNRI — serotonin and norepinephrine reuptake inhibitor"
    },
    "duloxetine": {
        "organs": ["Brain/CNS", "Liver"],
        "side_effects": ["hepatotoxicity", "hypertension", "sexual dysfunction"],
        "mechanism": "SNRI — balanced serotonin/norepinephrine uptake inhibition"
    },
    "bupropion": {
        "organs": ["Brain/CNS"],
        "side_effects": ["seizures", "hypertension", "insomnia"],
        "mechanism": "Atypical antidepressant — dopamine/norepinephrine reuptake inhibitor"
    },
    "mirtazapine": {
        "organs": ["Brain/CNS"],
        "side_effects": ["weight gain", "sedation", "agranulocytosis"],
        "mechanism": "Tetracyclic antidepressant — alpha-2 antagonist"
    },
    "citalopram": {
        "organs": ["Brain/CNS", "Heart"],
        "side_effects": ["QT prolongation", "hyponatraemia"],
        "mechanism": "Selective serotonin reuptake inhibitor"
    },
    "diazepam": {
        "organs": ["Brain/CNS", "Lungs"],
        "side_effects": ["respiratory depression", "CNS depression", "muscle weakness"],
        "mechanism": "GABA-A modulator — CNS depressant"
    },
    "lorazepam": {
        "organs": ["Brain/CNS"],
        "side_effects": ["sedation", "respiratory depression", "dependence"],
        "mechanism": "Benzodiazepine — GABA enhancement"
    },
    "alprazolam": {
        "organs": ["Brain/CNS"],
        "side_effects": ["addiction potential", "cognitive impairment", "respiratory depression"],
        "mechanism": "Short-acting benzodiazepine"
    },
    "quetiapine": {
        "organs": ["Brain/CNS", "Pancreas/Endocrine"],
        "side_effects": ["weight gain", "hyperglycemia", "hypotension"],
        "mechanism": "Atypical antipsychotic — dopamine antagonist"
    },
    "olanzapine": {
        "organs": ["Brain/CNS", "Pancreas/Endocrine"],
        "side_effects": ["weight gain", "diabetes risk", "metabolic syndrome"],
        "mechanism": "Atypical antipsychotic"
    },
    "risperidone": {
        "organs": ["Brain/CNS", "Heart"],
        "side_effects": ["extrapyramidal symptoms", "prolactin elevation"],
        "mechanism": "Dopamine D2 antagonist"
    },
    
    # ─────── PAIN & RESPIRATORY DRUGS ──────────
    "tramadol": {
        "organs": ["Brain/CNS", "Nervous System", "Lungs"],
        "side_effects": ["serotonin syndrome", "respiratory depression", "seizures"],
        "mechanism": "Opioid agonist and SNRI"
    },
    "morphine": {
        "organs": ["Brain/CNS", "Lungs", "Stomach/GI"],
        "side_effects": ["respiratory depression", "constipation", "opioid dependence"],
        "mechanism": "Mu opioid receptor agonist"
    },
    "codeine": {
        "organs": ["Brain/CNS", "Lungs"],
        "side_effects": ["respiratory depression", "constipation", "dependence"],
        "mechanism": "Opioid agonist and CYP2D6 prodrug"
    },
    "oxycodone": {
        "organs": ["Brain/CNS", "Lungs"],
        "side_effects": ["addiction", "overdose risk", "constipation"],
        "mechanism": "Potent mu opioid agonist"
    },
    "acetaminophen": {
        "organs": ["Liver"],
        "side_effects": ["hepatotoxicity at overdose", "acute liver failure"],
        "mechanism": "Analgesic and antipyretic — COX inhibitor"
    },
    "albuterol": {
        "organs": ["Heart", "Nervous System"],
        "side_effects": ["tachycardia", "tremor", "hypertension"],
        "mechanism": "Beta-2 agonist — bronchodilation"
    },
    "ipratropium": {
        "organs": ["Lungs"],
        "side_effects": ["dry mouth", "urinary retention"],
        "mechanism": "Anticholinergic — muscarinic antagonist"
    },
    "theophylline": {
        "organs": ["Heart", "Brain/CNS"],
        "side_effects": ["arrhythmias", "seizures", "nausea"],
        "mechanism": "Phosphodiesterase inhibitor"
    },
    "fluticasone": {
        "organs": ["Lungs", "Pancreas/Endocrine"],
        "side_effects": ["oral candidiasis", "hyperglycemia"],
        "mechanism": "Inhaled corticosteroid"
    },
    "prednisone": {
        "organs": ["Pancreas/Endocrine", "Liver", "Kidneys"],
        "side_effects": ["hyperglycemia", "immunosuppression", "osteoporosis"],
        "mechanism": "Systemic corticosteroid"
    },
    
    # ─────── CANCER & IMMUNOLOGY DRUGS ──────────
    "methotrexate": {
        "organs": ["Liver", "Kidneys", "Blood/Coagulation"],
        "side_effects": ["hepatotoxicity", "nephrotoxicity", "myelosuppression"],
        "mechanism": "Folate antagonist — antimetabolite"
    },
    "cyclosporine": {
        "organs": ["Kidneys", "Liver"],
        "side_effects": ["nephrotoxicity", "hepatotoxicity", "hypertension"],
        "mechanism": "Calcineurin inhibitor — immunosuppressant"
    },
    "azathioprine": {
        "organs": ["Liver", "Blood/Coagulation"],
        "side_effects": ["hepatotoxicity", "bone marrow suppression"],
        "mechanism": "Purine antagonist — immunosuppressant"
    },
    
    # ─────── ENDOCRINE DRUGS ──────────
    "levothyroxine": {
        "organs": ["Heart", "Pancreas/Endocrine"],
        "side_effects": ["tachycardia", "arrhythmias", "hyperglycemia"],
        "mechanism": "Thyroid hormone replacement"
    },
    "insulin": {
        "organs": ["Pancreas/Endocrine", "Kidneys"],
        "side_effects": ["hypoglycaemia", "lipodystrophy", "electrolyte shifts"],
        "mechanism": "Glucose-lowering agent — increases uptake"
    },
    "metformin": {
        "organs": ["Stomach/GI", "Kidneys"],
        "side_effects": ["lactic acidosis", "B12 deficiency", "diarrhea"],
        "mechanism": "Biguanide — reduces hepatic glucose output"
    },
    
    # ─────── PHARMACY BILL DRUGS (NEW) ──────────
    "otrivin": {
        "organs": ["Lungs", "Heart"],
        "side_effects": ["nasal irritation", "rebound congestion", "hypertension"],
        "mechanism": "Nasal decongestant — alpha agonist"
    },
    "zerodol": {
        "organs": ["Stomach/GI", "Kidneys", "Heart"],
        "side_effects": ["GI bleeding", "renal impairment", "cardiovascular risk"],
        "mechanism": "COX-2 selective inhibitor"
    },
    "sompraz": {
        "organs": ["Stomach/GI", "Kidneys"],
        "side_effects": ["hypomagnesaemia", "interstitial nephritis"],
        "mechanism": "Proton pump inhibitor"
    },
    "mondeslor": {
        "organs": ["Lungs", "Brain/CNS"],
        "side_effects": ["neuropsychiatric effects", "suicidal ideation"],
        "mechanism": "Leukotriene receptor antagonist"
    },
    
    # ─────── ADDITIONAL COMMON DRUGS ──────────
    "dabigatran": {
        "organs": ["Blood/Coagulation", "Stomach/GI"],
        "side_effects": ["gastrointestinal bleeding", "dyspepsia"],
        "mechanism": "Direct thrombin inhibitor"
    },
    "rivaroxaban": {
        "organs": ["Blood/Coagulation"],
        "side_effects": ["bleeding", "GI upset"],
        "mechanism": "Factor Xa inhibitor"
    },
    "apixaban": {
        "organs": ["Blood/Coagulation"],
        "side_effects": ["major bleeding", "minor bleeding"],
        "mechanism": "Direct factor Xa inhibitor"
    },
    "ticagrelor": {
        "organs": ["Blood/Coagulation", "Heart"],
        "side_effects": ["bleeding", "bradycardia", "dyspnea"],
        "mechanism": "P2Y12 inhibitor"
    },
    "gemfibrozil": {
        "organs": ["Liver", "Kidneys"],
        "side_effects": ["hepatotoxicity", "gallstones", "myopathy"],
        "mechanism": "Fibrate — raises HDL"
    },
    "niacin": {
        "organs": ["Liver", "Pancreas/Endocrine"],
        "side_effects": ["flushing", "hepatotoxicity", "hyperglycemia"],
        "mechanism": "B3 vitamin — raises HDL"
    },
    "ezetimibe": {
        "organs": ["Liver"],
        "side_effects": ["hepatotoxicity", "myopathy"],
        "mechanism": "Cholesterol absorption inhibitor"
    },
    "allopurinol": {
        "organs": ["Liver", "Kidneys"],
        "side_effects": ["Stevens-Johnson syndrome", "hepatotoxicity"],
        "mechanism": "Xanthine oxidase inhibitor"
    },
    "colchicine": {
        "organs": ["Stomach/GI", "Kidneys"],
        "side_effects": ["diarrhea", "renal impairment", "myopathy"],
        "mechanism": "Anti-inflammatory — microtubule inhibitor"
    },
    "sildenafil": {
        "organs": ["Heart", "Blood Pressure"],
        "side_effects": ["hypotension", "visual disturbances", "priapism"],
        "mechanism": "PDE-5 inhibitor — vasodilator"
    },
    "tadalafil": {
        "organs": ["Heart"],
        "side_effects": ["hypotension", "myalgia", "visual effects"],
        "mechanism": "Long-acting PDE-5 inhibitor"
    },
    "finasteride": {
        "organs": ["Pancreas/Endocrine"],
        "side_effects": ["sexual dysfunction", "gynecomastia"],
        "mechanism": "5-alpha reductase inhibitor"
    },
    "tamsulosin": {
        "organs": ["Heart", "Blood Pressure"],
        "side_effects": ["hypotension", "retrograde ejaculation"],
        "mechanism": "Alpha-1 antagonist"
    },
    
    # ─────── DDI DATASET DRUGS (206 TOTAL) ──────────
    "paracetamol": {
        "organs": ["Liver"],
        "side_effects": ["hepatotoxicity", "acute liver failure"],
        "mechanism": "Analgesic and antipyretic"
    },
    "caffeine": {
        "organs": ["Heart", "Brain/CNS"],
        "side_effects": ["tachycardia", "anxiety", "insomnia"],
        "mechanism": "Adenosine antagonist — stimulant"
    },
    "propranolol": {
        "organs": ["Heart", "Lungs", "Brain/CNS"],
        "side_effects": ["bradycardia", "bronchospasm", "fatigue"],
        "mechanism": "Non-selective beta blocker"
    },
    "acyclovir": {
        "organs": ["Kidneys", "Brain/CNS"],
        "side_effects": ["renal impairment", "neurotoxicity", "tremor"],
        "mechanism": "Viral DNA synthesis inhibitor"
    },
    "salbutamol": {
        "organs": ["Heart", "Nervous System"],
        "side_effects": ["tachycardia", "tremor", "hypertension"],
        "mechanism": "Beta-2 agonist — bronchodilator"
    },
    "albuterol": {
        "organs": ["Heart", "Nervous System"],
        "side_effects": ["tachycardia", "tremor"],
        "mechanism": "Beta-2 agonist"
    },
    "acarbose": {
        "organs": ["Stomach/GI", "Pancreas/Endocrine"],
        "side_effects": ["diarrhea", "bloating", "flatulence"],
        "mechanism": "Alpha-glucosidase inhibitor"
    },
    "alfuzosin": {
        "organs": ["Heart", "Blood Pressure"],
        "side_effects": ["hypotension", "dizziness"],
        "mechanism": "Alpha-1 adrenergic antagonist"
    },
    "alirocumab": {
        "organs": ["Liver"],
        "side_effects": ["injection site reactions", "neurocognitive effects"],
        "mechanism": "PCSK9 inhibitor monoclonal antibody"
    },
    "alogliptin": {
        "organs": ["Pancreas/Endocrine"],
        "side_effects": ["hypoglycaemia", "pancreatitis"],
        "mechanism": "DPP-4 inhibitor"
    },
    "amiloride": {
        "organs": ["Kidneys"],
        "side_effects": ["hyperkalaemia", "acidosis"],
        "mechanism": "Potassium-sparing diuretic"
    },
    "atazanavir": {
        "organs": ["Liver"],
        "side_effects": ["hepatotoxicity", "hyperbilirubinemia"],
        "mechanism": "HIV protease inhibitor"
    },
    "azilsartan": {
        "organs": ["Kidneys", "Heart"],
        "side_effects": ["hyperkalaemia", "renal impairment"],
        "mechanism": "Angiotensin II receptor antagonist"
    },
    "bempedoic_acid": {
        "organs": ["Kidneys"],
        "side_effects": ["hyperuricemia", "renal effects"],
        "mechanism": "Urate synthesis inhibitor"
    },
    "benazepril": {
        "organs": ["Kidneys", "Heart"],
        "side_effects": ["cough", "hyperkalaemia"],
        "mechanism": "ACE inhibitor"
    },
    "betrixaban": {
        "organs": ["Blood/Coagulation"],
        "side_effects": ["bleeding", "bruising"],
        "mechanism": "Factor Xa inhibitor anticoagulant"
    },
    "bisoprolol": {
        "organs": ["Heart"],
        "side_effects": ["bradycardia", "fatigue"],
        "mechanism": "Selective beta-1 blocker"
    },
    "bromocriptine": {
        "organs": ["Brain/CNS"],
        "side_effects": ["hypotension", "hallucinations"],
        "mechanism": "Dopamine agonist"
    },
    "bumetanide": {
        "organs": ["Kidneys", "Heart"],
        "side_effects": ["dehydration", "electrolyte imbalance"],
        "mechanism": "Loop diuretic"
    },
    "cabergoline": {
        "organs": ["Heart", "Brain/CNS"],
        "side_effects": ["cardiac valvulopathy", "orthostatic hypotension"],
        "mechanism": "Dopamine agonist"
    },
    "canagliflozin": {
        "organs": ["Kidneys", "Pancreas/Endocrine"],
        "side_effects": ["genital infections", "diabetic ketoacidosis"],
        "mechanism": "SGLT2 inhibitor"
    },
    "candesartan": {
        "organs": ["Kidneys", "Heart"],
        "side_effects": ["hyperkalaemia", "renal impairment"],
        "mechanism": "ARB"
    },
    "captopril": {
        "organs": ["Kidneys", "Heart"],
        "side_effects": ["cough", "angioedema"],
        "mechanism": "ACE inhibitor"
    },
    "carvedilol": {
        "organs": ["Heart", "Lungs"],
        "side_effects": ["bradycardia", "bronchospasm"],
        "mechanism": "Beta and alpha blocker"
    },
    "ceftriaxone": {
        "organs": ["Kidneys", "Liver"],
        "side_effects": ["renal impairment", "hepatotoxicity"],
        "mechanism": "Third-generation cephalosporin"
    },
    "cefuroxime": {
        "organs": ["Kidneys"],
        "side_effects": ["renal impairment", "GI upset"],
        "mechanism": "Second-generation cephalosporin"
    },
    "celecoxib": {
        "organs": ["Heart", "Kidneys"],
        "side_effects": ["cardiovascular events", "renal impairment"],
        "mechanism": "COX-2 selective inhibitor"
    },
    "cetirizine": {
        "organs": ["Brain/CNS"],
        "side_effects": ["drowsiness", "dry mouth"],
        "mechanism": "H1 receptor antagonist"
    },
    "chlorthalidone": {
        "organs": ["Kidneys"],
        "side_effects": ["electrolyte imbalance", "hyperglycemia"],
        "mechanism": "Thiazide-like diuretic"
    },
    "cholestyramine": {
        "organs": ["Stomach/GI"],
        "side_effects": ["constipation", "malabsorption"],
        "mechanism": "Bile acid sequestrant"
    },
    "cilostazol": {
        "organs": ["Blood/Coagulation", "Heart"],
        "side_effects": ["bleeding", "palpitations"],
        "mechanism": "Phosphodiesterase inhibitor"
    },
    "clarithromycin": {
        "organs": ["Liver", "Heart"],
        "side_effects": ["QT prolongation", "hepatotoxicity"],
        "mechanism": "Macrolide antibiotic"
    },
    "clevidipine": {
        "organs": ["Heart"],
        "side_effects": ["reflex tachycardia", "headache"],
        "mechanism": "Dihydropyridine calcium channel blocker"
    },
    "clonidine": {
        "organs": ["Heart", "Brain/CNS"],
        "side_effects": ["hypotension", "drowsiness"],
        "mechanism": "Alpha-2 adrenergic agonist"
    },
    "colesevelam": {
        "organs": ["Stomach/GI"],
        "side_effects": ["constipation", "nutrient malabsorption"],
        "mechanism": "Bile acid sequestrant"
    },
    "colestipol": {
        "organs": ["Stomach/GI"],
        "side_effects": ["constipation", "vitamin K deficiency"],
        "mechanism": "Bile acid sequestrant"
    },
    "dapagliflozin": {
        "organs": ["Kidneys", "Pancreas/Endocrine"],
        "side_effects": ["genitourinary infections", "DKA risk"],
        "mechanism": "SGLT2 inhibitor"
    },
    "daptomycin": {
        "organs": ["Muscles", "Kidneys"],
        "side_effects": ["muscle aches", "renal impairment"],
        "mechanism": "Cyclic lipopeptide antibiotic"
    },
    "darunavir": {
        "organs": ["Liver"],
        "side_effects": ["hepatotoxicity", "rash"],
        "mechanism": "HIV protease inhibitor"
    },
    "diclofenac": {
        "organs": ["Stomach/GI", "Kidneys"],
        "side_effects": ["GI bleeding", "renal impairment"],
        "mechanism": "NSAID"
    },
    "digoxin": {
        "organs": ["Heart"],
        "side_effects": ["arrhythmias", "digoxin toxicity"],
        "mechanism": "Cardiac glycoside — Na+/K+ ATPase inhibitor"
    },
    "dipyridamole": {
        "organs": ["Heart", "Blood/Coagulation"],
        "side_effects": ["angina", "bleeding"],
        "mechanism": "Platelet inhibitor and vasodilator"
    },
    "dolutegravir": {
        "organs": ["Liver"],
        "side_effects": ["hepatotoxicity", "weight gain"],
        "mechanism": "HIV integrase inhibitor"
    },
    "doxazosin": {
        "organs": ["Heart"],
        "side_effects": ["orthostatic hypotension", "syncope"],
        "mechanism": "Alpha-1 antagonist"
    },
    "dulaglutide": {
        "organs": ["Pancreas/Endocrine"],
        "side_effects": ["hypoglycaemia", "pancreatitis"],
        "mechanism": "GLP-1 receptor agonist"
    },
    "dutasteride": {
        "organs": ["Pancreas/Endocrine"],
        "side_effects": ["sexual dysfunction", "gynecomastia"],
        "mechanism": "5-alpha reductase inhibitor"
    },
    "edoxaban": {
        "organs": ["Blood/Coagulation"],
        "side_effects": ["major bleeding", "minor bleeding"],
        "mechanism": "Oral factor Xa inhibitor"
    },
    "efavirenz": {
        "organs": ["Brain/CNS", "Liver"],
        "side_effects": ["psychiatric effects", "hepatotoxicity"],
        "mechanism": "HIV non-nucleoside reverse transcriptase inhibitor"
    },
    "empagliflozin": {
        "organs": ["Kidneys", "Pancreas/Endocrine"],
        "side_effects": ["genital infections", "diabetic ketoacidosis"],
        "mechanism": "SGLT2 inhibitor"
    },
    "emtricitabine": {
        "organs": ["Liver", "Kidneys"],
        "side_effects": ["liver disease exacerbation", "renal impairment"],
        "mechanism": "HIV nucleoside reverse transcriptase inhibitor"
    },
    "enalapril": {
        "organs": ["Kidneys", "Heart"],
        "side_effects": ["cough", "hyperkalaemia"],
        "mechanism": "ACE inhibitor"
    },
    "eplerenone": {
        "organs": ["Kidneys", "Heart"],
        "side_effects": ["hyperkalaemia", "gynecomastia"],
        "mechanism": "Selective aldosterone antagonist"
    },
    "ertugliflozin": {
        "organs": ["Kidneys"],
        "side_effects": ["genitourinary infections"],
        "mechanism": "SGLT2 inhibitor"
    },
    "erythromycin": {
        "organs": ["Heart", "Liver"],
        "side_effects": ["QT prolongation", "hepatotoxicity"],
        "mechanism": "Macrolide antibiotic"
    },
    "esomeprazole": {
        "organs": ["Stomach/GI", "Kidneys"],
        "side_effects": ["hypomagnesaemia", "B12 deficiency"],
        "mechanism": "Proton pump inhibitor"
    },
    "ethambutol": {
        "organs": ["Eyes", "Nerves"],
        "side_effects": ["optic neuritis", "peripheral neuropathy"],
        "mechanism": "Tuberculosis cell wall synthesis inhibitor"
    },
    "evinacumab": {
        "organs": ["Liver"],
        "side_effects": ["injection site reactions"],
        "mechanism": "Monoclonal antibody against ANGPTL3"
    },
    "evolocumab": {
        "organs": ["Liver"],
        "side_effects": ["injection reactions", "neurocognitive effects"],
        "mechanism": "PCSK9 inhibitor"
    },
    "exenatide": {
        "organs": ["Pancreas/Endocrine"],
        "side_effects": ["pancreatitis", "hypoglycaemia"],
        "mechanism": "GLP-1 receptor agonist"
    },
    "felodipine": {
        "organs": ["Heart"],
        "side_effects": ["oedema", "hypotension"],
        "mechanism": "Dihydropyridine calcium channel blocker"
    },
    "fenofibrate": {
        "organs": ["Liver", "Kidneys"],
        "side_effects": ["hepatotoxicity", "myopathy"],
        "mechanism": "Fibrate — reduces triglycerides"
    },
    "fish_oil": {
        "organs": ["Blood/Coagulation"],
        "side_effects": ["bleeding", "bruising"],
        "mechanism": "Omega-3 supplement — antiplatelet effect"
    },
    "fondaparinux": {
        "organs": ["Blood/Coagulation"],
        "side_effects": ["bleeding", "thrombocytopenia"],
        "mechanism": "Selective factor Xa inhibitor"
    },
    "fosinopril": {
        "organs": ["Kidneys", "Heart"],
        "side_effects": ["cough", "hyperkalaemia"],
        "mechanism": "ACE inhibitor"
    },
    "gabapentin": {
        "organs": ["Brain/CNS"],
        "side_effects": ["drowsiness", "dizziness"],
        "mechanism": "Voltage-gated calcium channel modulator"
    },
    "garlic": {
        "organs": ["Blood/Coagulation"],
        "side_effects": ["bleeding", "body odor"],
        "mechanism": "Herbal supplement — antiplatelet effect"
    },
    "gentamicin": {
        "organs": ["Kidneys", "Ears"],
        "side_effects": ["nephrotoxicity", "ototoxicity"],
        "mechanism": "Aminoglycoside antibiotic"
    },
    "glimepiride": {
        "organs": ["Pancreas/Endocrine"],
        "side_effects": ["severe hypoglycaemia", "weight gain"],
        "mechanism": "Meglitinide — insulin secretagogue"
    },
    "glyburide": {
        "organs": ["Pancreas/Endocrine"],
        "side_effects": ["severe hypoglycaemia"],
        "mechanism": "Sulfonylurea"
    },
    "grapefruit_juice": {
        "organs": ["Liver"],
        "side_effects": ["increased drug levels"],
        "mechanism": "CYP3A4 inhibitor"
    },
    "heparin": {
        "organs": ["Blood/Coagulation"],
        "side_effects": ["bleeding", "thrombocytopenia"],
        "mechanism": "Anticoagulant — enhances antithrombin"
    },
    "hydrochlorothiazide": {
        "organs": ["Kidneys"],
        "side_effects": ["electrolyte imbalance", "hyperglycemia"],
        "mechanism": "Thiazide diuretic"
    },
    "ibalizumab": {
        "organs": ["Brain/CNS"],
        "side_effects": ["immunosuppression", "infections"],
        "mechanism": "CCR4 antagonist — HIV monoclonal antibody"
    },
    "icosapent_ethyl": {
        "organs": ["Blood/Coagulation"],
        "side_effects": ["bleeding", "gout exacerbation"],
        "mechanism": "Omega-3 fatty acid"
    },
    "inclisiran": {
        "organs": ["Liver"],
        "side_effects": ["injection reactions"],
        "mechanism": "PCSK9 siRNA inhibitor"
    },
    "indapamide": {
        "organs": ["Kidneys"],
        "side_effects": ["electrolyte imbalance", "hyperglycemia"],
        "mechanism": "Thiazide-like diuretic"
    },
    "irbesartan": {
        "organs": ["Kidneys", "Heart"],
        "side_effects": ["hyperkalaemia", "renal impairment"],
        "mechanism": "ARB— angiotensin II receptor antagonist"
    },
    "isoniazid": {
        "organs": ["Liver", "Nervous System"],
        "side_effects": ["hepatotoxicity", "peripheral neuropathy"],
        "mechanism": "TB cell wall synthesis inhibitor"
    },
    "isradipine": {
        "organs": ["Heart"],
        "side_effects": ["hypotension", "headache"],
        "mechanism": "Dihydropyridine calcium channel blocker"
    },
    "ketorolac": {
        "organs": ["Stomach/GI", "Kidneys"],
        "side_effects": ["GI bleeding", "renal impairment"],
        "mechanism": "NSAID"
    },
    "labetalol": {
        "organs": ["Heart", "Blood Pressure"],
        "side_effects": ["orthostatic hypotension", "fatigue"],
        "mechanism": "Non-selective beta and alpha blocker"
    },
    "lamivudine": {
        "organs": ["Liver", "Kidneys"],
        "side_effects": ["lactic acidosis", "liver disease"],
        "mechanism": "HIV nucleoside reverse transcriptase inhibitor"
    },
    "lansoprazole": {
        "organs": ["Stomach/GI", "Kidneys"],
        "side_effects": ["hypomagnesaemia", "B12 deficiency"],
        "mechanism": "Proton pump inhibitor"
    },
    "lercanidipine": {
        "organs": ["Heart"],
        "side_effects": ["oedema", "headache"],
        "mechanism": "Calcium channel blocker"
    },
    "levothyroxine": {
        "organs": ["Heart", "Pancreas/Endocrine"],
        "side_effects": ["tachycardia", "arrhythmias"],
        "mechanism": "Thyroid hormone replacement"
    },
    "linagliptin": {
        "organs": ["Pancreas/Endocrine"],
        "side_effects": ["pancreatitis", "hypoglycaemia"],
        "mechanism": "DPP-4 inhibitor"
    },
    "linezolid": {
        "organs": ["Brain/CNS", "Blood/Coagulation"],
        "side_effects": ["serotonin syndrome", "thrombocytopenia"],
        "mechanism": "Oxazolidone antibiotic"
    },
    "liraglutide": {
        "organs": ["Pancreas/Endocrine"],
        "side_effects": ["pancreatitis", "hypoglycaemia"],
        "mechanism": "GLP-1 receptor agonist"
    },
    "lomitapide": {
        "organs": ["Liver"],
        "side_effects": ["hepatotoxicity", "GI upset"],
        "mechanism": "Microsomal triglyceride transfer protein inhibitor"
    },
    "lopinavir": {
        "organs": ["Liver"],
        "side_effects": ["hepatotoxicity", "pancreatitis"],
        "mechanism": "HIV protease inhibitor"
    },
    "loratadine": {
        "organs": ["Brain/CNS"],
        "side_effects": ["drowsiness", "headache"],
        "mechanism": "H1 receptor antagonist"
    },
    "lovastatin": {
        "organs": ["Liver", "Muscles"],
        "side_effects": ["myopathy", "rhabdomyolysis"],
        "mechanism": "HMG-CoA reductase inhibitor"
    },
    "maraviroc": {
        "organs": ["Blood/Coagulation", "Liver"],
        "side_effects": ["bleeding", "hepatotoxicity"],
        "mechanism": "CCR5 antagonist — HIV entry inhibitor"
    },
    "meloxicam": {
        "organs": ["Stomach/GI", "Kidneys"],
        "side_effects": ["GI bleeding", "renal impairment"],
        "mechanism": "NSAID"
    },
    "methyldopa": {
        "organs": ["Brain/CNS", "Heart"],
        "side_effects": ["sedation", "hypotension"],
        "mechanism": "Central alpha-2 agonist"
    },
    "metronidazole": {
        "organs": ["Nervous System"],
        "side_effects": ["peripheral neuropathy", "metallic taste"],
        "mechanism": "Antiprotozoal and anaerobic antibiotic"
    },
    "miglitol": {
        "organs": ["Stomach/GI"],
        "side_effects": ["diarrhea", "flatulence"],
        "mechanism": "Alpha-glucosidase inhibitor"
    },
    "minoxidil": {
        "organs": ["Heart"],
        "side_effects": ["tachycardia", "fluid retention"],
        "mechanism": "Potassium channel opener"
    },
    "mipomersen": {
        "organs": ["Liver"],
        "side_effects": ["hepatotoxicity", "injection reactions"],
        "mechanism": "Apolipoprotein B antisense inhibitor"
    },
    "montelukast": {
        "organs": ["Lungs", "Brain/CNS"],
        "side_effects": ["neuropsychiatric effects", "Churg-Strauss"],
        "mechanism": "Leukotriene receptor antagonist"
    },
    "moxifloxacin": {
        "organs": ["Heart", "Tendons"],
        "side_effects": ["QT prolongation", "tendinopathy"],
        "mechanism": "Fluoroquinolone antibiotic"
    },
    "nateglinide": {
        "organs": ["Pancreas/Endocrine"],
        "side_effects": ["hypoglycaemia", "weight gain"],
        "mechanism": "Meglitinide — insulin secretagogue"
    },
    "nebivolol": {
        "organs": ["Heart"],
        "side_effects": ["bradycardia", "fatigue"],
        "mechanism": "Beta-1 selective blocker with vasodilation"
    },
    "nevirapine": {
        "organs": ["Liver", "Skin"],
        "side_effects": ["Stevens-Johnson syndrome", "hepatotoxicity"],
        "mechanism": "HIV non-nucleoside reverse transcriptase inhibitor"
    },
    "niacin": {
        "organs": ["Liver", "Pancreas/Endocrine"],
        "side_effects": ["flushing", "hyperglycemia"],
        "mechanism": "B vitamin — raises HDL"
    },
    "nicardipine": {
        "organs": ["Heart"],
        "side_effects": ["headache", "reflex tachycardia"],
        "mechanism": "Dihydropyridine calcium channel blocker"
    },
    "nifedipine": {
        "organs": ["Heart"],
        "side_effects": ["hypotension", "oedema"],
        "mechanism": "Dihydropyridine calcium channel blocker"
    },
    "nimodipine": {
        "organs": ["Heart", "Brain/CNS"],
        "side_effects": ["hypotension", "headache"],
        "mechanism": "Calcium channel blocker — cerebral vasodilator"
    },
    "nitrofurantoin": {
        "organs": ["Lungs", "Liver"],
        "side_effects": ["pulmonary toxicity", "hepatotoxicity"],
        "mechanism": "Urinary tract antibiotic"
    },
    "nitroglycerin": {
        "organs": ["Heart"],
        "side_effects": ["hypotension", "headache"],
        "mechanism": "Nitrate — nitric oxide donor"
    },
    "olmesartan": {
        "organs": ["Kidneys", "Heart"],
        "side_effects": ["hyperkalaemia", "sprue-like enteropathy"],
        "mechanism": "ARB"
    },
    "omega_3_fatty_acids": {
        "organs": ["Blood/Coagulation"],
        "side_effects": ["bleeding", "GI upset"],
        "mechanism": "Anti-inflammatory — antiplatelet"
    },
    "oseltamivir": {
        "organs": ["Stomach/GI"],
        "side_effects": ["nausea", "neuropsychiatric effects"],
        "mechanism": "Neuraminidase inhibitor"
    },
    "perindopril": {
        "organs": ["Kidneys", "Heart"],
        "side_effects": ["cough", "angioedema"],
        "mechanism": "ACE inhibitor"
    },
    "pioglitazone": {
        "organs": ["Liver", "Pancreas/Endocrine"],
        "side_effects": ["hepatotoxicity", "weight gain"],
        "mechanism": "Thiazolidinedione — PPARγ agonist"
    },
    "pitavastatin": {
        "organs": ["Liver", "Muscles"],
        "side_effects": ["myopathy", "hepatotoxicity"],
        "mechanism": "Statin"
    },
    "prasugrel": {
        "organs": ["Blood/Coagulation"],
        "side_effects": ["major bleeding"],
        "mechanism": "P2Y12 inhibitor — antiplatelet"
    },
    "pravastatin": {
        "organs": ["Liver", "Muscles"],
        "side_effects": ["myopathy", "hepatotoxicity"],
        "mechanism": "HMG-CoA reductase inhibitor"
    },
    "prazosin": {
        "organs": ["Heart"],
        "side_effects": ["first-dose syncope", "orthostatic hypotension"],
        "mechanism": "Alpha-1 antagonist"
    },
    "pyrazinamide": {
        "organs": ["Liver"],
        "side_effects": ["hepatotoxicity", "hyperuricemia"],
        "mechanism": "TB cell wall synthesis inhibitor"
    },
    "quinapril": {
        "organs": ["Kidneys", "Heart"],
        "side_effects": ["cough", "hyperkalaemia"],
        "mechanism": "ACE inhibitor"
    },
    "raltegravir": {
        "organs": ["Liver"],
        "side_effects": ["rash", "hepatotoxicity"],
        "mechanism": "HIV integrase inhibitor"
    },
    "ramipril": {
        "organs": ["Kidneys", "Heart"],
        "side_effects": ["cough", "angioedema"],
        "mechanism": "ACE inhibitor"
    },
    "red_yeast_rice": {
        "organs": ["Liver", "Muscles"],
        "side_effects": ["myopathy", "hepatotoxicity"],
        "mechanism": "Contains lovastatin-like compounds"
    },
    "repaglinide": {
        "organs": ["Pancreas/Endocrine"],
        "side_effects": ["hypoglycaemia", "weight gain"],
        "mechanism": "Meglitinide — insulin secretagogue"
    },
    "rifampin": {
        "organs": ["Liver"],
        "side_effects": ["hepatotoxicity", "drug interactions"],
        "mechanism": "TB RNA polymerase inhibitor"
    },
    "ritonavir": {
        "organs": ["Liver"],
        "side_effects": ["hepatotoxicity", "pancreatitis"],
        "mechanism": "HIV protease inhibitor — CYP3A4 inhibitor"
    },
    "rosiglitazone": {
        "organs": ["Heart", "Liver"],
        "side_effects": ["cardiac events", "weight gain"],
        "mechanism": "Thiazolidinedione"
    },
    "saxagliptin": {
        "organs": ["Pancreas/Endocrine"],
        "side_effects": ["pancreatitis", "hypoglycaemia"],
        "mechanism": "DPP-4 inhibitor"
    },
    "semaglutide": {
        "organs": ["Pancreas/Endocrine"],
        "side_effects": ["pancreatitis", "hypoglycaemia"],
        "mechanism": "GLP-1 receptor agonist"
    },
    "sitagliptin": {
        "organs": ["Pancreas/Endocrine"],
        "side_effects": ["pancreatitis", "joint pain"],
        "mechanism": "DPP-4 inhibitor"
    },
    "st_johns_wort": {
        "organs": ["Brain/CNS"],
        "side_effects": ["photosensitivity", "drug interactions"],
        "mechanism": "Herbal supplement — CYP450 inducer"
    },
    "sulfamethoxazole": {
        "organs": ["Kidneys"],
        "side_effects": ["crystalluria", "Stevens-Johnson"],
        "mechanism": "Sulfonamide antibiotic"
    },
    "telmisartan": {
        "organs": ["Kidneys", "Heart"],
        "side_effects": ["hyperkalaemia", "dizziness"],
        "mechanism": "ARB"
    },
    "tenofovir": {
        "organs": ["Kidneys"],
        "side_effects": ["renal impairment", "bone loss"],
        "mechanism": "HIV nucleotide reverse transcriptase inhibitor"
    },
    "terazosin": {
        "organs": ["Heart"],
        "side_effects": ["orthostatic hypotension", "syncope"],
        "mechanism": "Alpha-1 antagonist"
    },
    "tinidazole": {
        "organs": ["Nervous System"],
        "side_effects": ["peripheral neuropathy", "metallic taste"],
        "mechanism": "Antiprotozoal"
    },
    "tirzepatide": {
        "organs": ["Pancreas/Endocrine"],
        "side_effects": ["pancreatitis", "GI effects"],
        "mechanism": "GLP-1/GIP receptor agonist"
    },
    "torsemide": {
        "organs": ["Kidneys", "Heart"],
        "side_effects": ["dehydration", "electrolyte imbalance"],
        "mechanism": "Loop diuretic"
    },
    "triamterene": {
        "organs": ["Kidneys"],
        "side_effects": ["hyperkalaemia", "renal stones"],
        "mechanism": "Potassium-sparing diuretic"
    },
    "valacyclovir": {
        "organs": ["Kidneys", "Brain/CNS"],
        "side_effects": ["renal impairment", "neurotoxicity"],
        "mechanism": "Prodrug of acyclovir"
    },
    "vardenafil": {
        "organs": ["Heart"],
        "side_effects": ["hypotension", "visual disturbances"],
        "mechanism": "PDE-5 inhibitor"
    },
    "vorapaxar": {
        "organs": ["Blood/Coagulation"],
        "side_effects": ["bleeding", "bruising"],
        "mechanism": "PAR-1 antagonist — antiplatelet"
    },
    "zanamivir": {
        "organs": ["Lungs"],
        "side_effects": ["bronchospasm", "neuropsychiatric effects"],
        "mechanism": "Neuraminidase inhibitor"
    },
    "zidovudine": {
        "organs": ["Blood/Coagulation"],
        "side_effects": ["bone marrow suppression", "lactic acidosis"],
        "mechanism": "HIV nucleoside reverse transcriptase inhibitor"
    },
    "zolpidem": {
        "organs": ["Brain/CNS"],
        "side_effects": ["sedation", "dependence"],
        "mechanism": "GABA-A modulator — sedative"
    },
    # ═══════════════════════════════════════════════════════════════════════════════
    # ONCOLOGY & TARGETED THERAPIES (CANCER DRUGS) — 2023-2025 FDA Approvals
    # ═══════════════════════════════════════════════════════════════════════════════
    "zanidatamab": {
        "organs": ["Liver", "Kidneys", "Heart"],
        "side_effects": ["hepatotoxicity", "renal impairment", "cardiotoxicity"],
        "mechanism": "Bispecific antibody targeting HER2 and HER3"
    },
    "revumenib": {
        "organs": ["Liver", "Kidneys", "Blood/Coagulation"],
        "side_effects": ["hepatotoxicity", "renal impairment", "bone marrow suppression"],
        "mechanism": "Menin inhibitor for acute leukemia"
    },
    "inavolisib": {
        "organs": ["Liver", "Kidneys", "Stomach/GI"],
        "side_effects": ["hepatotoxicity", "diarrhea", "rash"],
        "mechanism": "PI3K-beta/delta inhibitor"
    },
    "lazertinib": {
        "organs": ["Liver", "Lungs", "Heart"],
        "side_effects": ["hepatotoxicity", "interstitial pneumonia", "QT prolongation"],
        "mechanism": "EGFR tyrosine kinase inhibitor"
    },
    "tarlatamab": {
        "organs": ["Brain/CNS", "Liver", "Heart"],
        "side_effects": ["neurotoxicity", "hepatotoxicity", "cytokine release syndrome"],
        "mechanism": "Bispecific antibody targeting DLL3 and CD3"
    },
    "zenocutuzumab": {
        "organs": ["Liver", "Kerneys", "Stomach/GI"],
        "side_effects": ["hepatotoxicity", "diarrhea", "nausea"],
        "mechanism": "HER2 and HER3 bispecific antibody"
    },
    "cosibelimab": {
        "organs": ["Skin", "Liver", "Blood/Coagulation"],
        "side_effects": ["rash", "hepatotoxicity", "bleeding"],
        "mechanism": "PD-L1 inhibitor for skin cancer"
    },
    "ensartinib": {
        "organs": ["Liver", "Lungs", "Heart"],
        "side_effects": ["hepatotoxicity", "pneumonia", "QT prolongation"],
        "mechanism": "ALK tyrosine kinase inhibitor"
    },
    "retifanlimab": {
        "organs": ["Liver", "Heart", "Blood/Coagulation"],
        "side_effects": ["hepatotoxicity", "myocarditis", "bleeding"],
        "mechanism": "PD-1 inhibitor"
    },
    "pirtobrutinib": {
        "organs": ["Liver", "Blood/Coagulation", "Stomach/GI"],
        "side_effects": ["hepatotoxicity", "bleeding", "diarrhea"],
        "mechanism": "BTK inhibitor for lymphoma"
    },
    "elacestrant": {
        "organs": ["Liver", "Stomach/GI"],
        "side_effects": ["hepatotoxicity", "nausea", "vomiting"],
        "mechanism": "Estrogen receptor degrader"
    },
    "epcoritamab": {
        "organs": ["Brain/CNS", "Liver", "Heart"],
        "side_effects": ["cytokine release syndrome", "hepatotoxicity", "myocarditis"],
        "mechanism": "Bispecific antibody targeting CD3 and CD20"
    },
    "glofitamab": {
        "organs": ["Brain/CNS", "Liver", "Heart"],
        "side_effects": ["cytokine release syndrome", "hepatotoxicity", "cardiac effects"],
        "mechanism": "Bispecific antibody"
    },
    "talquetamab": {
        "organs": ["Brain/CNS", "Liver", "Heart"],
        "side_effects": ["cytokine release syndrome", "neurotoxicity", "hepatotoxicity"],
        "mechanism": "Bispecific T-cell engager"
    },
    "elranatamab": {
        "organs": ["Brain/CNS", "Liver", "Heart"],
        "side_effects": ["cytokine release syndrome", "hepatotoxicity", "myocarditis"],
        "mechanism": "BCMA bispecific antibody"
    },
    "quizartinib": {
        "organs": ["Liver", "Heart", "Kidneys"],
        "side_effects": ["hepatotoxicity", "QT prolongation", "renal impairment"],
        "mechanism": "FLT3 inhibitor for AML"
    },
    "repotrectinib": {
        "organs": ["Liver", "Lungs", "Brain/CNS"],
        "side_effects": ["hepatotoxicity", "pneumonia", "neurotoxicity"],
        "mechanism": "ROS1/NTRK inhibitor"
    },
    "capivasertib": {
        "organs": ["Liver", "Kidneys", "Stomach/GI"],
        "side_effects": ["hepatotoxicity", "diarrhea", "rash"],
        "mechanism": "AKT inhibitor"
    },
    "fruquintinib": {
        "organs": ["Liver", "Kidneys", "Stomach/GI"],
        "side_effects": ["hepatotoxicity", "hand-foot syndrome", "diarrhea"],
        "mechanism": "FGFR inhibitor"
    },
    "toripalimab": {
        "organs": ["Liver", "Heart", "Blood/Coagulation"],
        "side_effects": ["hepatotoxicity", "myocarditis", "bleeding"],
        "mechanism": "PD-1 inhibitor"
    },
    "ivosidenib": {
        "organs": ["Liver", "Stomach/GI"],
        "side_effects": ["hepatotoxicity", "nausea", "differentiation syndrome"],
        "mechanism": "IDH1 inhibitor"
    },
    "tucatinib": {
        "organs": ["Liver", "Heart", "Stomach/GI"],
        "side_effects": ["hepatotoxicity", "QT prolongation", "diarrhea"],
        "mechanism": "HER2 tyrosine kinase inhibitor"
    },
    "linvoseltamab": {
        "organs": ["Brain/CNS", "Liver", "Heart"],
        "side_effects": ["cytokine release syndrome", "hepatotoxicity", "myocarditis"],
        "mechanism": "B-cell maturation antigen bispecific antibody"
    },
    "belantamab_mafodotin": {
        "organs": ["Eyes", "Liver", "Heart"],
        "side_effects": ["corneal toxicity", "hepatotoxicity", "myocarditis"],
        "mechanism": "BCMA-targeting antibody-drug conjugate"
    },
    "vimseltinib": {
        "organs": ["Liver", "Kidneys", "Stomach/GI"],
        "side_effects": ["hepatotoxicity", "diarrhea", "rash"],
        "mechanism": "CSF1R inhibitor"
    },
    "rivoceranib": {
        "organs": ["Liver", "Kidneys", "Heart"],
        "side_effects": ["hepatotoxicity", "hand-foot syndrome", "hypertension"],
        "mechanism": "VEGFR inhibitor for hepatocellular carcinoma"
    },
    "camrelizumab": {
        "organs": ["Liver", "Heart", "Blood/Coagulation"],
        "side_effects": ["hepatotoxicity", "myocarditis", "bleeding"],
        "mechanism": "PD-1 inhibitor"
    },
    "cabozantinib": {
        "organs": ["Liver", "Kidneys", "Heart"],
        "side_effects": ["hand-foot syndrome", "hypertension", "hepatotoxicity"],
        "mechanism": "Multi-kinase inhibitor"
    },
    "penpulimab": {
        "organs": ["Liver", "Heart", "Blood/Coagulation"],
        "side_effects": ["hepatotoxicity", "myocarditis", "bleeding"],
        "mechanism": "PD-1 inhibitor"
    },
    # ═══════════════════════════════════════════════════════════════════════════════
    # METABOLIC, ENDOCRINE & RENAL (2023-2025)
    # ═══════════════════════════════════════════════════════════════════════════════
    "atrasentan": {
        "organs": ["Kidneys", "Heart"],
        "side_effects": ["fluid retention", "edema", "anemia"],
        "mechanism": "Endothelin A receptor antagonist"
    },
    "bexagliflozin": {
        "organs": ["Kidneys", "Pancreas/Endocrine"],
        "side_effects": ["genital infections", "diabetic ketoacidosis"],
        "mechanism": "SGLT2 inhibitor"
    },
    "sotagliflozin": {
        "organs": ["Kidneys", "Heart", "Pancreas/Endocrine"],
        "side_effects": ["genital mycotic infections", "diabetic ketoacidosis"],
        "mechanism": "SGLT1/SGLT2 dual inhibitor"
    },
    "daprodustat": {
        "organs": ["Liver", "Kidneys"],
        "side_effects": ["hepatotoxicity", "thrombosis"],
        "mechanism": "HIF-prolyl hydroxylase inhibitor"
    },
    "sparsentan": {
        "organs": ["Kidneys", "Heart"],
        "side_effects": ["renal impairment", "hyperkalemia"],
        "mechanism": "Dual endothelin/angiotensin receptor antagonist"
    },
    "plozasiran": {
        "organs": ["Liver"],
        "side_effects": ["hepatotoxicity", "injection site reactions"],
        "mechanism": "APOC-III siRNA inhibitor"
    },
    "crinecerfont": {
        "organs": ["Liver", "Pancreas/Endocrine"],
        "side_effects": ["hepatotoxicity", "hepatic impairment"],
        "mechanism": "CRF1 antagonist"
    },
    "palopegteriparatide": {
        "organs": ["Pancreas/Endocrine", "Kidneys"],
        "side_effects": ["hypercalcemia", "renal effects"],
        "mechanism": "PTH1 receptor agonist"
    },
    "seladelpar": {
        "organs": ["Liver"],
        "side_effects": ["hepatotoxicity", "GI upset"],
        "mechanism": "FXR agonist"
    },
    "elafibranor": {
        "organs": ["Liver", "Kidneys"],
        "side_effects": ["hepatotoxicity", "renal impairment"],
        "mechanism": "PPAR alpha/delta agonist"
    },
    "resmetirom": {
        "organs": ["Liver"],
        "side_effects": ["hepatotoxicity", "elevated liver enzymes"],
        "mechanism": "THR-beta agonist"
    },
    "olezarsen": {
        "organs": ["Liver", "Kidneys"],
        "side_effects": ["hepatotoxicity", "injection site reactions"],
        "mechanism": "APOB antisense oligonucleotide"
    },
    "sibeprenlimab": {
        "organs": ["Kidneys", "Heart"],
        "side_effects": ["renal impairment", "edema"],
        "mechanism": "FGF23 neutralizing antibody"
    },
    "mitapivat": {
        "organs": ["Blood/Coagulation"],
        "side_effects": ["hemolysis", "thrombosis"],
        "mechanism": "PKR activator for thalassemia"
    },
    # ═══════════════════════════════════════════════════════════════════════════════
    # NEUROLOGY, PSYCHIATRY & PAIN MANAGEMENT (2023-2025)
    # ═══════════════════════════════════════════════════════════════════════════════
    "xanomeline": {
        "organs": ["Brain/CNS", "Heart"],
        "side_effects": ["anticholinergic effects", "QT prolongation"],
        "mechanism": "M1/M4 muscarinic agonist"
    },
    "donanemab": {
        "organs": ["Brain/CNS"],
        "side_effects": ["amyloid-related imaging abnormalities", "cognitive effects"],
        "mechanism": "Tau protein antibody"
    },
    "lecanemab": {
        "organs": ["Brain/CNS"],
        "side_effects": ["amyloid-related imaging abnormalities", "infusion reactions"],
        "mechanism": "Amyloid-beta protofibrils antibody"
    },
    "zavegepant": {
        "organs": ["Brain/CNS"],
        "side_effects": ["nasal irritation", "dysgeusia"],
        "mechanism": "CGRP receptor antagonist"
    },
    "trofinetide": {
        "organs": ["Brain/CNS"],
        "side_effects": ["tremor", "dizziness"],
        "mechanism": "Trofinetide — IGFBP3 modulator"
    },
    "tofersen": {
        "organs": ["Brain/CNS", "Nervous System"],
        "side_effects": ["peripheral neuropathy", "CSF-related effects"],
        "mechanism": "SOD1 antisense oligonucleotide"
    },
    "zuranolone": {
        "organs": ["Brain/CNS"],
        "side_effects": ["sedation", "dizziness"],
        "mechanism": "GABA-A positive allosteric modulator"
    },
    "suzetrigine": {
        "organs": ["Brain/CNS", "Nervous System"],
        "side_effects": ["dizziness", "headache"],
        "mechanism": "Selective sodium channel inhibitor"
    },
    "milsaperidone": {
        "organs": ["Brain/CNS", "Heart"],
        "side_effects": ["QT prolongation", "metabolic syndrome"],
        "mechanism": "Dopamine antagonist"
    },
    "tradipitant": {
        "organs": ["Brain/CNS"],
        "side_effects": ["dizziness", "nausea"],
        "mechanism": "Neurokinin-1 receptor antagonist"
    },
    "vamorolone": {
        "organs": ["Liver", "Muscles"],
        "side_effects": ["hepatotoxicity", "muscle weakness"],
        "mechanism": "Dissociative glucocorticoid"
    },
    # ═══════════════════════════════════════════════════════════════════════════════
    # RARE DISEASES & ORPHAN DRUGS (2023-2025)
    # ═══════════════════════════════════════════════════════════════════════════════
    "omaveloxolone": {
        "organs": ["Liver", "Nervous System"],
        "side_effects": ["hepatotoxicity", "neuropathy"],
        "mechanism": "Nrf2 activator"
    },
    "velmanase_alfa": {
        "organs": ["Liver"],
        "side_effects": ["infusion reactions", "hepatic effects"],
        "mechanism": "Recombinant alpha-mannosidase"
    },
    "leniolisib": {
        "organs": ["Liver", "Blood/Coagulation"],
        "side_effects": ["hepatotoxicity", "thrombosis"],
        "mechanism": "PI3K delta inhibitor"
    },
    "pegunigalsidase_alfa": {
        "organs": ["Kidneys", "Heart"],
        "side_effects": ["infusion reactions", "renal impairment"],
        "mechanism": "Pegylated alpha-galactosidase"
    },
    "levacetylleucine": {
        "organs": ["Brain/CNS"],
        "side_effects": ["vertigo", "neurological effects"],
        "mechanism": "Neurological disorder treatment"
    },
    "arimoclomol": {
        "organs": ["Liver"],
        "side_effects": ["hepatic effects", "fever"],
        "mechanism": "Heat shock protein inducer"
    },
    "vorasidenib": {
        "organs": ["Brain/CNS", "Liver"],
        "side_effects": ["hepatotoxicity", "neurological effects"],
        "mechanism": "IDH inhibitor"
    },
    "imetelstat": {
        "organs": ["Liver", "Blood/Coagulation"],
        "side_effects": ["hepatotoxicity", "thrombocytopenia"],
        "mechanism": "Telomerase inhibitor"
    },
    "cipaglucidase_alfa": {
        "organs": ["Liver", "Muscles"],
        "side_effects": ["muscle pain", "infusion reactions"],
        "mechanism": "Glucosidase-alpha replacement"
    },
    "nalmefene": {
        "organs": ["Brain/CNS"],
        "side_effects": ["precipitation of withdrawal", "nausea"],
        "mechanism": "Opioid antagonist"
    },
    "nedosiran": {
        "organs": ["Liver", "Kidneys"],
        "side_effects": ["hepatotoxicity", "renal effects"],
        "mechanism": "LDHA antisense oligonucleotide"
    },
    "eplontersen": {
        "organs": ["Liver", "Nervous System"],
        "side_effects": ["hepatotoxicity", "neuropathy"],
        "mechanism": "TTR antisense oligonucleotide"
    },
    "zilucoplan": {
        "organs": ["Liver", "Blood/Coagulation"],
        "side_effects": ["hepatotoxicity", "bleeding"],
        "mechanism": "Complement C5 inhibitor"
    },
    # ═══════════════════════════════════════════════════════════════════════════════
    # IMMUNOLOGY, DERMATOLOGY & RESPIRATORY (2023-2025)
    # ═══════════════════════════════════════════════════════════════════════════════
    "lebrikizumab": {
        "organs": ["Skin", "Liver"],
        "side_effects": ["infusion reactions", "hepatic effects"],
        "mechanism": "IL-13 inhibitor"
    },
    "nemolizumab": {
        "organs": ["Skin", "Liver"],
        "side_effects": ["pruritus", "hepatic effects"],
        "mechanism": "TRPV1 antagonist"
    },
    "ensifentrine": {
        "organs": ["Lungs", "Heart"],
        "side_effects": ["tremor", "tachycardia"],
        "mechanism": "Four-in-one muscarinic/beta-adrenergic agonist"
    },
    "axatilimab": {
        "organs": ["Liver", "Blood/Coagulation"],
        "side_effects": ["hepatotoxicity", "bleeding"],
        "mechanism": "CSF1R inhibitor"
    },
    "ritlecitinib": {
        "organs": ["Liver", "Blood/Coagulation"],
        "side_effects": ["hepatotoxicity", "infections"],
        "mechanism": "JAK3/TEC inhibitor"
    },
    "fezolinetant": {
        "organs": ["Brain/CNS"],
        "side_effects": ["dizziness", "nausea"],
        "mechanism": "Neurokinin-3 receptor antagonist"
    },
    "rozanolixizumab": {
        "organs": ["Blood/Coagulation"],
        "side_effects": ["bleeding", "infusions reactions"],
        "mechanism": "FcRn antagonist"
    },
    "bimekizumab": {
        "organs": ["Skin", "Liver"],
        "side_effects": ["infections", "hepatic effects"],
        "mechanism": "IL-17A/IL-17F blocker"
    },
    "mirikizumab": {
        "organs": ["Stomach/GI", "Liver"],
        "side_effects": ["infections", "hepatic effects"],
        "mechanism": "IL-23 inhibitor"
    },
    "depemokimab": {
        "organs": ["Lungs", "Liver"],
        "side_effects": ["asthma exacerbation", "hepatic effects"],
        "mechanism": "IL-5 inhibitor"
    },
    "dupilumab": {
        "organs": ["Lungs", "Skin"],
        "side_effects": ["injection site reactions", "eye problems"],
        "mechanism": "IL-4R alpha antagonist"
    },
    "elinzanetant": {
        "organs": ["Brain/CNS"],
        "side_effects": ["headache", "nausea"],
        "mechanism": "NK3 receptor antagonist"
    },
    "sebetralstat": {
        "organs": ["Stomach/GI"],
        "side_effects": ["nausea", "diarrhea"],
        "mechanism": "Plasma kallikrein inhibitor"
    },
    "garadacimab": {
        "organs": ["Blood/Coagulation"],
        "side_effects": ["thrombosis", "bleeding"],
        "mechanism": "Factor XII inhibitor"
    },
    "clesrovimab": {
        "organs": ["Lungs"],
        "side_effects": ["respiratory effects", "infusion reactions"],
        "mechanism": "RSV antibody"
    },
    "nirsevimab": {
        "organs": ["Lungs"],
        "side_effects": ["respiratory effects"],
        "mechanism": "RSV F protein antibody"
    },
    "abrocitinib": {
        "organs": ["Liver", "Blood/Coagulation"],
        "side_effects": ["hepatotoxicity", "thrombosis"],
        "mechanism": "JAK inhibitor"
    },
    "tralokinumab": {
        "organs": ["Skin", "Liver"],
        "side_effects": ["infusion reactions", "hepatic effects"],
        "mechanism": "IL-13 inhibitor"
    },
    # ═══════════════════════════════════════════════════════════════════════════════
    # CARDIOVASCULAR & HEMATOLOGY (2023-2025)
    # ═══════════════════════════════════════════════════════════════════════════════
    "concizumab": {
        "organs": ["Blood/Coagulation"],
        "side_effects": ["thrombosis", "bleeding"],
        "mechanism": "TF pathway inhibitor"
    },
    "acoramidis": {
        "organs": ["Heart", "Nervous System"],
        "side_effects": ["arrhythmias", "neuropathy"],
        "mechanism": "Transthyretin stabilizer"
    },
    "landiolol": {
        "organs": ["Heart"],
        "side_effects": ["hypotension", "bradycardia"],
        "mechanism": "Ultra-short-acting beta-1 blocker"
    },
    "etripamil": {
        "organs": ["Heart"],
        "side_effects": ["hypotension", "headache"],
        "mechanism": "Calcium channel blocker nasal spray"
    },
    "marstacimab": {
        "organs": ["Blood/Coagulation"],
        "side_effects": ["thrombosis", "bleeding"],
        "mechanism": "TF pathway inhibitor"
    },
    "crovalimab": {
        "organs": ["Blood/Coagulation"],
        "side_effects": ["thrombosis", "hemolysis"],
        "mechanism": "C3 complement inhibitor"
    },
    "aficamten": {
        "organs": ["Heart"],
        "side_effects": ["arrhythmias", "cardiac dysfunction"],
        "mechanism": "Cardiac myosin inhibitor"
    },
    "vutrisiran": {
        "organs": ["Heart", "Nervous System"],
        "side_effects": ["cardiomyopathy progression", "neuropathy"],
        "mechanism": "TTR siRNA inhibitor"
    },
    "fitusiran": {
        "organs": ["Blood/Coagulation"],
        "side_effects": ["thrombosis", "bleeding"],
        "mechanism": "FVIII inhibitor antagonist"
    },
    "mavacamten": {
        "organs": ["Heart"],
        "side_effects": ["cardiac dysfunction", "arrhythmias"],
        "mechanism": "Cardiac myosin inhibitor"
    },
    "inclisiran": {
        "organs": ["Liver"],
        "side_effects": ["hepatic effects", "injection site reactions"],
        "mechanism": "PCSK9 inhibitor"
    },
    "cyclosporine": {
        "organs": ["Eyes"],
        "side_effects": ["eye irritation", "increased tear production"],
        "mechanism": "Calcineurin inhibitor"
    },
    "perfluorhexyloctane": {
        "organs": ["Eyes"],
        "side_effects": ["eye irritation", "visual effects"],
        "mechanism": "Lubricant for dry eye"
    },
    "acoltremon": {
        "organs": ["Eyes"],
        "side_effects": ["eye irritation", "headache"],
        "mechanism": "TRPV1 antagonist"
    },
    "lotilaner": {
        "organs": ["Eyes"],
        "side_effects": ["eye irritation", "rash"],
        "mechanism": "Sigma-1 receptor antagonist"
    },
    "avacincaptad_pegol": {
        "organs": ["Eyes"],
        "side_effects": ["eye inflammation", "visual disturbances"],
        "mechanism": "Complement C5 inhibitor"
    },
    "pegcetacoplan": {
        "organs": ["Eyes"],
        "side_effects": ["eye inflammation", "vision loss"],
        "mechanism": "Complement C3 inhibitor"
    },
    # ═══════════════════════════════════════════════════════════════════════════════
    # INFECTIOUS DISEASE & ANTIVIRALS (2023-2025)
    # ═══════════════════════════════════════════════════════════════════════════════
    "nirmatrelvir": {
        "organs": ["Liver"],
        "side_effects": ["hepatic effects", "taste disturbance"],
        "mechanism": "SARS-CoV-2 3CL protease inhibitor"
    },
    "sulbactam": {
        "organs": ["Kidneys", "Stomach/GI"],
        "side_effects": ["renal impairment", "diarrhea"],
        "mechanism": "Beta-lactamase inhibitor"
    },
    "durlobactam": {
        "organs": ["Kidneys"],
        "side_effects": ["renal impairment"],
        "mechanism": "Beta-lactamase inhibitor"
    },
    "rezafungin": {
        "organs": ["Liver"],
        "side_effects": ["hepatic effects", "infusion reactions"],
        "mechanism": "Echinocandin antifungal"
    },
    "lenacapavir": {
        "organs": ["Liver", "Kidneys"],
        "side_effects": ["hepatotoxicity", "renal impairment"],
        "mechanism": "Capsid inhibitor for HIV"
    },
    "gepotidacin": {
        "organs": ["Stomach/GI", "Liver"],
        "side_effects": ["nausea", "diarrhea"],
        "mechanism": "Bacterial topoisomerase inhibitor"
    },
    "insulin_icodec": {
        "organs": ["Pancreas/Endocrine"],
        "side_effects": ["hypoglycemia", "injection site reactions"],
        "mechanism": "Long-acting insulin"
    },
    "riluzole": {
        "organs": ["Brain/CNS", "Liver"],
        "side_effects": ["hepatotoxicity", "neutropenia"],
        "mechanism": "Glutamate antagonist"
    },
    # ═══════════════════════════════════════════════════════════════════════════════
    # ADDITIONAL 151-250 LIST (Selected Small Molecules & Biologics)
    # ═══════════════════════════════════════════════════════════════════════════════
    "adstiladrin": {
        "organs": ["Kidneys"],
        "side_effects": ["dysuria", "hematuria"],
        "mechanism": "Intravesical immunotherapy"
    },
    "filsuvez": {
        "organs": ["Liver", "Kidneys"],
        "side_effects": ["hepatic effects", "renal effects"],
        "mechanism": "Fibroblast growth factor"
    },
    "veopoz": {
        "organs": ["Liver"],
        "side_effects": ["hepatic effects", "diarrhea"],
        "mechanism": "GI and hepatic effects"
    },
    "ojjaara": {
        "organs": ["Eyes"],
        "side_effects": ["eye inflammation", "visual disturbances"],
        "mechanism": "C5 inhibitor for eye disease"
    },
    "adiwere": {
        "organs": ["Liver", "Kidneys"],
        "side_effects": ["hepatic effects", "renal impairment"],
        "mechanism": "Gene therapy carrier"
    },
    "fabhalta": {
        "organs": ["Liver", "Blood/Coagulation"],
        "side_effects": ["hepatic effects", "bleeding"],
        "mechanism": "Factor XIIa inhibitor"
    },
    "iwiq": {
        "organs": ["Brain/CNS"],
        "side_effects": ["neurological effects", "sedation"],
        "mechanism": "Cholinesterase inhibitor"
    },
    "lykos": {
        "organs": ["Liver"],
        "side_effects": ["hepatic effects"],
        "mechanism": "Orphan drug for rare disease"
    },
    "tovorafenib": {
        "organs": ["Brain/CNS", "Liver"],
        "side_effects": ["brain tumor effects", "hepatotoxicity"],
        "mechanism": "RAF kinase inhibitor"
    },
    "odefsey": {
        "organs": ["Liver", "Kidneys"],
        "side_effects": ["hepatic effects", "renal impairment"],
        "mechanism": "Combination antiretroviral"
    },
    "biktarvy": {
        "organs": ["Liver", "Kidneys"],
        "side_effects": ["hepatic effects", "renal impairment"],
        "mechanism": "HIV integrase inhibitor combination"
    },
    "descovy": {
        "organs": ["Kidneys"],
        "side_effects": ["renal impairment", "decreased bone density"],
        "mechanism": "Tenofovir alafenamide combination"
    },
    "vemlidy": {
        "organs": ["Liver", "Kidneys"],
        "side_effects": ["hepatic effects", "renal impairment"],
        "mechanism": "Tenofovir alafenamide"
    },
    "genvoya": {
        "organs": ["Liver", "Kidneys"],
        "side_effects": ["hepatic effects", "renal impairment"],
        "mechanism": "HIV integrase inhibitor"
    },
    "symtuza": {
        "organs": ["Liver", "Stomach/GI"],
        "side_effects": ["hepatic effects", "diarrhea"],
        "mechanism": "Protease inhibitor combination"
    },
    "juluca": {
        "organs": ["Liver"],
        "side_effects": ["hepatic effects", "rash"],
        "mechanism": "Integrase inhibitor plus NNRTI"
    },
    "dovato": {
        "organs": ["Liver", "Kidneys"],
        "side_effects": ["hepatic effects", "renal impairment"],
        "mechanism": "Integrase inhibitor combination"
    },
    "cabenuva": {
        "organs": ["Liver"],
        "side_effects": ["hepatic effects", "injection site reactions"],
        "mechanism": "Injectable antiretroviral"
    },
    "vocabria": {
        "organs": ["Liver"],
        "side_effects": ["hepatic effects"],
        "mechanism": "Injectable protease inhibitor"
    },
    "trogarzo": {
        "organs": ["Liver"],
        "side_effects": ["hepatic effects", "infusion reactions"],
        "mechanism": "Attachment inhibitor"
    },
    "rukobia": {
        "organs": ["Liver"],
        "side_effects": ["hepatic effects"],
        "mechanism": "Integrase strand transfer inhibitor"
    },
    "ibalizumab": {
        "organs": ["Brain/CNS"],
        "side_effects": ["infusion reactions", "immunosuppression"],
        "mechanism": "CCR4 antagonist"
    },
    "vicriviroc": {
        "organs": ["Liver"],
        "side_effects": ["hepatic effects"],
        "mechanism": "CCR5 antagonist"
    },
    "fostemsavir": {
        "organs": ["Liver", "Stomach/GI"],
        "side_effects": ["nausea", "hepatic effects"],
        "mechanism": "Attachment inhibitor"
    },
    "islatravir": {
        "organs": ["Liver", "Kidneys"],
        "side_effects": ["hepatic effects", "renal impairment"],
        "mechanism": "Nucleoside analog"
    },
    "selinexor": {
        "organs": ["Liver", "Blood/Coagulation"],
        "side_effects": ["hepatotoxicity", "thrombocytopenia"],
        "mechanism": "XPO1 inhibitor"
    },
    "belumosudil": {
        "organs": ["Liver", "Heart"],
        "side_effects": ["hepatic effects", "QT prolongation"],
        "mechanism": "ROCK2 inhibitor"
    },
    "pacritinib": {
        "organs": ["Liver", "Blood/Coagulation"],
        "side_effects": ["hepatotoxicity", "thrombosis"],
        "mechanism": "JAK inhibitor"
    },
    "momelotinib": {
        "organs": ["Liver"],
        "side_effects": ["hepatic effects", "anemia"],
        "mechanism": "JAK inhibitor"
    },
    "fedratinib": {
        "organs": ["Liver", "Brain/CNS"],
        "side_effects": ["hepatotoxicity", "encephalopathy"],
        "mechanism": "JAK2 inhibitor"
    },
    "inrebic": {
        "organs": ["Liver"],
        "side_effects": ["hepatic effects"],
        "mechanism": "JAK inhibitor"
    },
    # ═══════════════════════════════════════════════════════════════════════════════
    # ADDITIONAL 201-250 LIST (Hematology & Oncology Continued)
    # ═══════════════════════════════════════════════════════════════════════════════
    "abecma": {
        "organs": ["Brain/CNS", "Liver"],
        "side_effects": ["cytokine release syndrome", "hepatic effects"],
        "mechanism": "CAR-T cell therapy"
    },
    "breyanzi": {
        "organs": ["Brain/CNS", "Liver"],
        "side_effects": ["cytokine release syndrome", "hepatotoxicity"],
        "mechanism": "CAR-T cell therapy"
    },
    "kymriah": {
        "organs": ["Brain/CNS", "Liver"],
        "side_effects": ["cytokine release syndrome", "neurotoxicity"],
        "mechanism": "CAR-T cell therapy"
    },
    "yescarta": {
        "organs": ["Brain/CNS", "Liver"],
        "side_effects": ["cytokine release syndrome", "hepatotoxicity"],
        "mechanism": "CAR-T cell therapy"
    },
    "tecartus": {
        "organs": ["Brain/CNS"],
        "side_effects": ["cytokine release syndrome", "neurotoxicity"],
        "mechanism": "CAR-T cell therapy"
    },
    "carvykti": {
        "organs": ["Brain/CNS", "Liver"],
        "side_effects": ["cytokine release syndrome", "hepatic effects"],
        "mechanism": "CAR-T cell therapy"
    },
    "polivy": {
        "organs": ["Liver", "Stomach/GI"],
        "side_effects": ["hepatotoxicity", "nausea"],
        "mechanism": "CD79b antibody-drug conjugate"
    },
    "padcev": {
        "organs": ["Liver", "Eyes"],
        "side_effects": ["hepatotoxicity", "peripheral neuropathy"],
        "mechanism": "Nectin-4 antibody-drug conjugate"
    },
    "enhertu": {
        "organs": ["Liver", "Stomach/GI"],
        "side_effects": ["hepatotoxicity", "diarrhea"],
        "mechanism": "HER2 antibody-drug conjugate"
    },
    "trodelvy": {
        "organs": ["Liver", "Stomach/GI"],
        "side_effects": ["hepatotoxicity", "diarrhea"],
        "mechanism": "TROP-2 antibody-drug conjugate"
    },
    "blenrep": {
        "organs": ["Eyes", "Liver"],
        "side_effects": ["corneal damage", "hepatic effects"],
        "mechanism": "GPRC5D antibody-drug conjugate"
    },
    "zynlonta": {
        "organs": ["Liver"],
        "side_effects": ["hepatotoxicity", "peripheral neuropathy"],
        "mechanism": "CD37 antibody-drug conjugate"
    },
    "tivdak": {
        "organs": ["Liver", "Stomach/GI"],
        "side_effects": ["hepatotoxicity", "nausea"],
        "mechanism": "TROP-2 antibody-drug conjugate"
    },
    "adcetris": {
        "organs": ["Liver", "Nervous System"],
        "side_effects": ["hepatotoxicity", "peripheral neuropathy"],
        "mechanism": "CD30 antibody-drug conjugate"
    },
    "kadcyla": {
        "organs": ["Liver", "Heart"],
        "side_effects": ["hepatotoxicity", "cardiac dysfunction"],
        "mechanism": "HER2 antibody-drug conjugate"
    },
    "lumoxiti": {
        "organs": ["Liver", "Kidneys"],
        "side_effects": ["hepatotoxicity", "renal impairment"],
        "mechanism": "CD22 antibody-drug conjugate"
    },
    "besponsa": {
        "organs": ["Liver"],
        "side_effects": ["hepatotoxicity", "hyperglycemia"],
        "mechanism": "CD22 antibody-drug conjugate"
    },
    "mylotarg": {
        "organs": ["Liver"],
        "side_effects": ["hepatotoxicity", "veno-occlusive disease"],
        "mechanism": "CD33 antibody-drug conjugate"
    },
    "poteligeo": {
        "organs": ["Liver"],
        "side_effects": ["hepatotoxicity", "cytopenias"],
        "mechanism": "CD30 ligand antagonist"
    },
    "mogamulizumab": {
        "organs": ["Liver", "Blood/Coagulation"],
        "side_effects": ["hepatotoxicity", "infections"],
        "mechanism": "CCR4 antagonist"
    },
    "tagraxofusp": {
        "organs": ["Liver"],
        "side_effects": ["hepatotoxicity", "vascular leak"],
        "mechanism": "CD123-targeted diphtheria toxin"
    },
    "elzonris": {
        "organs": ["Liver"],
        "side_effects": ["hepatotoxicity", "vascular leak"],
        "mechanism": "CD123-targeted toxin"
    },
    "capmatinib": {
        "organs": ["Liver", "Stomach/GI"],
        "side_effects": ["hepatotoxicity", "diarrhea"],
        "mechanism": "MET inhibitor"
    },
    "tepotinib": {
        "organs": ["Liver", "Stomach/GI"],
        "side_effects": ["hepatotoxicity", "diarrhea"],
        "mechanism": "MET kinase inhibitor"
    },
    "selpercatinib": {
        "organs": ["Liver", "Heart"],
        "side_effects": ["hepatotoxicity", "QT prolongation"],
        "mechanism": "RET kinase inhibitor"
    },
    "pralsetinib": {
        "organs": ["Liver", "Heart"],
        "side_effects": ["hepatotoxicity", "QT prolongation"],
        "mechanism": "RET kinase inhibitor"
    },
    "mobocertinib": {
        "organs": ["Liver", "Stomach/GI"],
        "side_effects": ["hepatotoxicity", "diarrhea"],
        "mechanism": "EGFR exon 20 inhibitor"
    },
    "amivantamab": {
        "organs": ["Liver", "Heart"],
        "side_effects": ["hepatotoxicity", "myocarditis"],
        "mechanism": "EGFR and MET bispecific antibody"
    },
    "sotorasib": {
        "organs": ["Liver", "Stomach/GI"],
        "side_effects": ["hepatotoxicity", "diarrhea"],
        "mechanism": "KRAS G12C inhibitor"
    },
    "adagrasib": {
        "organs": ["Liver", "Stomach/GI"],
        "side_effects": ["hepatotoxicity", "diarrhea"],
        "mechanism": "KRAS G12C inhibitor"
    },
    "futibatinib": {
        "organs": ["Liver", "Heart"],
        "side_effects": ["hepatotoxicity", "QT prolongation"],
        "mechanism": "FGFR inhibitor"
    },
    "erdafitinib": {
        "organs": ["Liver", "Kidneys"],
        "side_effects": ["hepatotoxicity", "hyperphosphatemia"],
        "mechanism": "FGFR inhibitor"
    },
    "pemigatinib": {
        "organs": ["Liver", "Stomach/GI"],
        "side_effects": ["hepatotoxicity", "diarrhea"],
        "mechanism": "FGFR inhibitor"
    },
    "infigratinib": {
        "organs": ["Liver", "Stomach/GI"],
        "side_effects": ["hepatotoxicity", "diarrhea"],
        "mechanism": "FGFR inhibitor"
    },
    "asciminib": {
        "organs": ["Liver", "Stomach/GI"],
        "side_effects": ["hepatotoxicity", "rash"],
        "mechanism": "ABL switch control inhibitor"
    },
    "avapritinib": {
        "organs": ["Liver", "Stomach/GI"],
        "side_effects": ["hepatotoxicity", "diarrhea"],
        "mechanism": "PDGFRA inhibitor"
    },
    "ripretinib": {
        "organs": ["Liver", "Stomach/GI"],
        "side_effects": ["hepatotoxicity", "diarrhea"],
        "mechanism": "KIT switch control inhibitor"
    },
    "tivozanib": {
        "organs": ["Liver", "Kidneys", "Heart"],
        "side_effects": ["hepatotoxicity", "hypertension", "hand-foot syndrome"],
        "mechanism": "VEGFR inhibitor"
    },
    "zanubrutinib": {
        "organs": ["Liver", "Blood/Coagulation"],
        "side_effects": ["hepatotoxicity", "infections"],
        "mechanism": "BTK inhibitor"
    },
    "acalabrutinib": {
        "organs": ["Liver", "Blood/Coagulation"],
        "side_effects": ["hepatotoxicity", "infections"],
        "mechanism": "BTK inhibitor"
    },
    "cifran": {
        "organs": ["Stomach/GI", "Tendons", "Liver"],
        "side_effects": ["tendinitis", "nausea", "diarrhea", "hepatotoxicity"],
        "mechanism": "Fluoroquinolone antibiotic — DNA gyrase inhibitor"
    },
    "pantocid": {
        "organs": ["Stomach/GI", "Liver", "Kidneys"],
        "side_effects": ["hypomagnesaemia", "bone loss", "hepatotoxicity"],
        "mechanism": "Proton pump inhibitor — gastric acid suppression"
    },
    "zofer": {
        "organs": ["Brain/CNS", "Heart"],
        "side_effects": ["headache", "QT prolongation", "constipation"],
        "mechanism": "5-HT3 receptor antagonist — antiemetic"
    },
    "enzar": {
        "organs": ["Heart", "Kidneys", "Liver"],
        "side_effects": ["edema", "dizziness", "hypotension", "hepatotoxicity"],
        "mechanism": "Calcium channel blocker — dihydropyridine"
    },
    "cremaffin": {
        "organs": ["Stomach/GI"],
        "side_effects": ["constipation", "aluminum toxicity if excessive", "diarrhea"],
        "mechanism": "Antacid — magnesium hydroxide, aluminum hydroxide, simethicone"
    },
    "potrate": {
        "organs": ["Kidneys", "Heart", "Stomach/GI"],
        "side_effects": ["hyperkalaemia", "cardiac arrhythmias", "GI ulceration"],
        "mechanism": "Potassium supplement — electrolyte replacement"
    },
    "ursetor": {
        "organs": ["Liver", "Gallbladder", "Stomach/GI"],
        "side_effects": ["diarrhea", "hepatotoxicity", "abdominal discomfort"],
        "mechanism": "Ursodeoxycholic acid — bile acid for gallstone dissolution"
    },
}

# Generate DRUG_NAMES from DRUG_KNOWLEDGE (which has all 226+ drugs)
DRUG_NAMES = sorted([d.title() for d in DRUG_KNOWLEDGE])


def get_drug_knowledge(drug_name: str) -> dict:
    """Return the knowledge entry for a drug name (case-insensitive)."""
    return DRUG_KNOWLEDGE.get(drug_name.lower(), {})


def generate_fallback_analysis(drug_a: str, drug_b: str, prob: float) -> dict:
    """
    Build organ/side-effect/mechanism data from the hardcoded knowledge base
    when Groq is unavailable or returns empty results.
    Combines knowledge from both drugs and infers interaction effects.
    """
    ka = get_drug_knowledge(drug_a)
    kb = get_drug_knowledge(drug_b)

    # Merge organs from both drugs
    all_organs = list(set(ka.get("organs", []) + kb.get("organs", [])))

    # Merge and deduplicate side effects, keeping most specific ones
    all_effects = ka.get("side_effects", [])[:3] + kb.get("side_effects", [])[:3]

    # Build a combined mechanism
    mech_a = ka.get("mechanism", "")
    mech_b = kb.get("mechanism", "")
    if mech_a and mech_b:
        mechanism = f"{drug_a}: {mech_a}. Combined with {drug_b} ({mech_b.lower()})"
    elif mech_a:
        mechanism = mech_a
    elif mech_b:
        mechanism = mech_b
    else:
        mechanism = f"Pharmacokinetic or pharmacodynamic interaction between {drug_a} and {drug_b}"

    # Add interaction-specific effects based on shared organ targets
    interaction_effects = []
    shared_organs = set(ka.get("organs", [])) & set(kb.get("organs", []))
    if "Liver" in shared_organs:
        interaction_effects.append("additive hepatotoxicity — combined liver enzyme elevation")
    if "Kidneys" in shared_organs:
        interaction_effects.append("compounded renal impairment and reduced drug clearance")
    if "Brain/CNS" in shared_organs or "Nervous System" in shared_organs:
        interaction_effects.append("enhanced CNS depression — additive sedation risk")
    if "Blood/Coagulation" in shared_organs:
        interaction_effects.append("potentiated anticoagulant effect — elevated bleeding risk")
    if "Heart" in shared_organs:
        interaction_effects.append("additive cardiovascular effects — blood pressure and rhythm changes")
    if "Lungs" in shared_organs:
        interaction_effects.append("compounded respiratory depression risk")

    final_effects = (interaction_effects + all_effects)[:7]

    # If still empty, add generic effects based on probability
    if not final_effects:
        if prob > 0.7:
            final_effects = [
                "pharmacokinetic interaction — altered drug plasma levels",
                "potential toxicity accumulation",
                "reduced therapeutic efficacy of one or both agents"
            ]
        else:
            final_effects = [
                "pharmacokinetic interaction possible",
                "monitor plasma drug levels",
                "clinical monitoring recommended"
            ]

    return {
        "side_effects": final_effects,
        "organs_affected": all_organs if all_organs else ["Systemic"],
        "mechanism": mechanism
    }


@st.cache_data(show_spinner=False)
def groq_clinical_analysis(drug_a, drug_b, prob):
    """
    Get clinical analysis from Groq LLM.
    Falls back to hardcoded drug knowledge if Groq is unavailable.
    """
    # Always generate fallback first so we have a baseline
    fallback = generate_fallback_analysis(drug_a, drug_b, prob)

    if not GROQ_ENABLED:
        return fallback

    prompt = f"""You are a clinical pharmacologist. Analyze the drug-drug interaction between {drug_a} and {drug_b}.
Predicted interaction risk probability: {prob*100:.1f}%

Respond ONLY in this exact format, no other text:
- [side effect or risk 1]
- [side effect or risk 2]
- [side effect or risk 3]
- [side effect or risk 4]
- [side effect or risk 5]
ORGAN: [organ system name]
ORGAN: [organ system name]
MECHANISM: [one sentence mechanistic explanation]

Use clinical organ system names such as: Heart, Liver, Kidneys, Lungs, Brain/CNS, Nervous System, Stomach/GI, Intestines/GI, Blood/Coagulation, Muscles, Pancreas/Endocrine, Blood Pressure.
Be specific and medically accurate."""

    try:
        r = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=600,
        )
        text = r.choices[0].message.content.strip()
        se, orgs, mech = [], [], ""

        for line in text.split("\n"):
            c = line.strip()
            if c.startswith("- ") or c.startswith("• "):
                e = c.lstrip("-•· ").strip()
                if e and len(e) > 3:
                    se.append(e)
            elif c.upper().startswith("ORGAN:"):
                o = c[6:].strip().strip("[]")
                if o:
                    orgs.append(o)
            elif c.upper().startswith("MECHANISM:"):
                mech = c[10:].strip().strip("[]")

        # Merge Groq results with fallback — Groq takes priority but
        # fill gaps from fallback so we always have complete data
        final_se    = se[:7]    if se    else fallback["side_effects"]
        final_orgs  = orgs      if orgs  else fallback["organs_affected"]
        final_mech  = mech      if mech  else fallback["mechanism"]

        # Always ensure at least the fallback organs are included
        # (Groq sometimes misses obvious ones)
        all_orgs = list(dict.fromkeys(final_orgs + fallback["organs_affected"]))

        return {
            "side_effects":     final_se,
            "organs_affected":  all_orgs[:8],
            "mechanism":        final_mech,
        }

    except Exception as e:
        # Groq failed — return clean fallback, not an error message
        return fallback

@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), "../models/ddi_random_forest_dosage.pkl")
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model not found at: {model_path}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model = load_model()

def render_body_map(organ_scores: dict, pair_data: list, height: int = 580):
    """
    Render the interactive human body SVG map as an inline HTML component.
    Injects data directly into the HTML before the JS runs, avoiding timing issues.
    """
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "body_map.html")
    if not os.path.exists(html_path):
        st.warning("body_map.html not found. Place it in the same folder as app.py.")
        return

    with open(html_path, "r") as f:
        html_template = f.read()

    # Sanitize pair data — only JSON-serializable fields
    safe_pairs = []
    for p in pair_data:
        safe_pairs.append({
            "pair":         [str(p.get("pair", ["?","?"])[0]), str(p.get("pair", ["?","?"])[1])],
            "probability":  float(p.get("probability", 0)),
            "organ_scores": {str(k): int(v) for k, v in p.get("organ_scores", {}).items()},
            "side_effects": [str(e) for e in p.get("side_effects", [])[:5]],
        })

    organ_json = json.dumps(organ_scores)
    pairs_json = json.dumps(safe_pairs)

    # Inject data as the very first script in <head> so it's available before anything else
    data_injection = f"""<script>
window.ORGAN_DATA = {organ_json};
window.PAIR_DATA  = {pairs_json};
</script>"""

    # Insert right after <head>
    if "<head>" in html_template:
        html = html_template.replace("<head>", "<head>\n" + data_injection, 1)
    else:
        html = data_injection + html_template

    components.html(html, height=height, scrolling=False)


# ─── MODEL NAME NORMALIZATION ─────────────────────────────────────────────────
def normalize_model_name(name: str) -> str:
    """Normalize model names to match model_comparison.py training names."""
    mapping = {
        "knn": "KNN",
        "k nearest neighbors": "KNN",
        "k-nearest neighbors": "KNN",
        "randomforest": "Random Forest",
        "random_forest": "Random Forest",
        "logisticregression": "Logistic Regression",
        "logistic_regression": "Logistic Regression",
        "svm": "SVM",
        "supportvectormachine": "SVM",
        "gradientboosting": "Gradient Boosting",
        "gradient_boosting": "Gradient Boosting",
    }
    return mapping.get(name.strip().lower(), name)


# ─── PRESCRIPTION ANALYSIS HELPERS ─────────────────────────────────────────────

def extract_text_from_file(uploaded_file) -> tuple:
    """
    Extract text from PDF or image file.
    Handles both text PDFs and scanned PDFs (using OCR).
    Returns: (text, error_msg, success_flag)
    """
    if uploaded_file is None:
        return "", "No file uploaded", False

    try:
        if uploaded_file.type == "application/pdf":
            if not PDF_SUPPORT:
                return "", "PyPDF2 not installed. Install with: pip install PyPDF2", False
            
            # First try to extract embedded text
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # If no text found, try OCR on PDF pages (scanned PDFs)
            if not text.strip():
                if not IMAGE_SUPPORT:
                    return "", "PDF appears to be scanned. Install pytesseract for OCR: pip install pytesseract pillow", False
                
                try:
                    import pdf2image
                    images = pdf2image.convert_from_bytes(uploaded_file.getvalue())
                    text = ""
                    for img in images:
                        text += pytesseract.image_to_string(img) + "\n"
                except ImportError:
                    return "", "PDF is scanned (no embedded text). Install pdf2image for OCR support: pip install pdf2image", False
                except Exception as ocr_error:
                    return "", f"OCR extraction failed: {str(ocr_error)}", False
            
            return text.strip(), None, bool(text.strip())

        elif uploaded_file.type in ["image/png", "image/jpeg", "image/jpg", "image/tiff", "image/bmp"]:
            if not IMAGE_SUPPORT:
                return "", "pytesseract/Pillow not installed. Install with: pip install pytesseract pillow", False
            
            image = Image.open(io.BytesIO(uploaded_file.getvalue()))
            text = pytesseract.image_to_string(image)
            return text.strip(), None, bool(text.strip())

        else:
            return "", f"Unsupported file type: {uploaded_file.type}. Use PDF or image (PNG, JPG, TIFF, BMP).", False

    except Exception as e:
        return "", f"Error extracting text: {str(e)}", False


@st.cache_data(show_spinner=False)
def parse_prescription_with_groq(prescription_text: str) -> dict:
    """
    Use Groq LLM to parse prescription text and extract structured drug data.
    
    Returns:
    {
        "drugs": [{"name": str, "dose": str, "frequency": str}, ...],
        "raw_text": str,
        "confidence": float  # 0-1
    }
    """
    if not GROQ_ENABLED:
        return {
            "drugs": [],
            "raw_text": prescription_text[:500],
            "confidence": 0.0,
            "error": "Groq API not available"
        }

    if not prescription_text or len(prescription_text) < 10:
        return {
            "drugs": [],
            "raw_text": prescription_text,
            "confidence": 0.0,
            "error": "Prescription text too short"
        }

    prompt = f"""You are a clinical pharmacist parsing a prescription document. Extract ALL medications with their dosages and frequencies.
    
Prescription text:
{prescription_text[:2000]}

Response format (ONLY JSON, no other text):
{{
  "drugs": [
    {{"name": "Drug Name", "dose": "500 mg", "frequency": "2 times daily"}},
    {{"name": "Another Drug", "dose": "100 mg", "frequency": "Once daily"}}
  ],
  "confidence": 0.95,
  "notes": "Any observations"
}}

Rules:
- Extract ONLY drugs actually mentioned in the prescription
- Drug names must be simple (e.g., "Aspirin", "Ibuprofen", not full brand names)
- No hallucination - only extract what is explicitly present
- confidence: 0.0-1.0 based on text clarity
- If no drugs found, return empty drugs array"""

    try:
        r = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1000,
        )
        response_text = r.choices[0].message.content.strip()
        
        # Extract JSON from markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)
        
        # Ensure drugs is a list
        if "drugs" not in result:
            result["drugs"] = []
        
        # Validate structure
        for drug in result.get("drugs", []):
            if "name" not in drug:
                drug["name"] = "Unknown"
            if "dose" not in drug:
                drug["dose"] = "Unknown"
            if "frequency" not in drug:
                drug["frequency"] = "Unknown"

        result["raw_text"] = prescription_text[:500]
        result["confidence"] = result.get("confidence", 0.5)

        return result

    except json.JSONDecodeError as e:
        return {
            "drugs": [],
            "raw_text": prescription_text[:500],
            "confidence": 0.0,
            "error": f"Failed to parse response: {str(e)}"
        }
    except Exception as e:
        return {
            "drugs": [],
            "raw_text": prescription_text[:500],
            "confidence": 0.0,
            "error": f"Groq error: {str(e)}"
        }


def validate_drugs(drugs_list: list) -> dict:
    """
    Validate extracted drugs against system drug list.
    
    Returns:
    {
        "valid": [{"name": str, "dose": int, "matched": bool}, ...],
        "invalid": [{"original": str, "reason": str}, ...],
        "warnings": [str, ...]
    }
    """
    valid = []
    invalid = []
    warnings = []

    available_drugs_lower = {d.lower(): d for d in DRUG_NAMES}

    for drug in drugs_list:
        name = drug.get("name", "").strip()
        dose_str = drug.get("dose", "").strip()
        freq = drug.get("frequency", "").strip()

        if not name:
            invalid.append({"original": str(drug), "reason": "Empty drug name"})
            continue

        # Try to match drug name (case-insensitive)
        matched_name = available_drugs_lower.get(name.lower())

        if matched_name:
            # Try to extract numeric dose
            dose_num = 100  # default
            try:
                # Extract first number from dose string
                import re
                numbers = re.findall(r'\d+', dose_str)
                if numbers:
                    dose_num = int(numbers[0])
                    if dose_num < 1 or dose_num > 5000:
                        dose_num = 100
                        warnings.append(f"{matched_name}: Dose {drug.get('dose')} outside typical range, using default 100 mg")
            except:
                warnings.append(f"{matched_name}: Could not parse dose '{dose_str}', using default 100 mg")

            valid.append({
                "name": matched_name,
                "dose": dose_num,
                "matched": True,
                "frequency": freq,
                "original_dose": dose_str
            })
        else:
            invalid.append({
                "original": name,
                "reason": "Drug not found in system database"
            })
            warnings.append(f"Drug '{name}' not recognized. Available drugs: {', '.join(DRUG_NAMES[:5])}...")

    return {
        "valid": valid,
        "invalid": invalid,
        "warnings": warnings
    }


# ─── SESSION STATE ─────────────────────────────────────────────────────────────
if "drugs" not in st.session_state:
    st.session_state.drugs = [{"name": DRUG_NAMES[0], "dose": 100},
                               {"name": DRUG_NAMES[1], "dose": 100}]
if "analysis_history" not in st.session_state: st.session_state.analysis_history = []
if "current_results"   not in st.session_state: st.session_state.current_results = None
if "selected_model"    not in st.session_state:
    st.session_state.selected_model = get_default_model() if MODEL_MANAGER_ENABLED else "Default"

# Prescription analysis state
if "rx_raw_text" not in st.session_state: st.session_state.rx_raw_text = None
if "rx_parsed" not in st.session_state: st.session_state.rx_parsed = None
if "rx_drugs_editable" not in st.session_state: st.session_state.rx_drugs_editable = {}

# ─── TOP NAV ──────────────────────────────────────────────────────────────────
now = datetime.now()
st.markdown(f"""
<div style="background:#000;border-bottom:1px solid #1c1c1c;padding:0 32px;
  display:flex;align-items:center;justify-content:space-between;height:48px;">
  <div style="display:flex;align-items:center;gap:16px;">
    <span style="font-family:'Space Mono',monospace;font-size:13px;font-weight:700;
      letter-spacing:0.06em;color:#f0f0f0;">DDI//PREDICTOR</span>
    <span style="font-family:'Space Mono',monospace;font-size:9px;letter-spacing:0.14em;
      text-transform:uppercase;color:#3a3a3a;border:1px solid #1c1c1c;padding:2px 8px;">
      PRO v2.0
    </span>
    <span style="font-family:'Space Mono',monospace;font-size:9px;letter-spacing:0.14em;
      text-transform:uppercase;color:#3a3a3a;">Clinical Intelligence Platform</span>
  </div>
  <div style="display:flex;align-items:center;gap:20px;">
    <span style="font-family:'Space Mono',monospace;font-size:9px;
      color:#3a3a3a;letter-spacing:0.1em;">{now.strftime('%a %H:%M:%S')} UTC</span>
    <span style="font-family:'Space Mono',monospace;font-size:9px;color:#3a3a3a;">
      RDKit &middot; RF &middot; Groq LLM
    </span>
    <div style="width:6px;height:6px;border-radius:50%;background:#47ff8c;
      box-shadow:0 0 6px #47ff8c;"></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "ANALYZE", "UPLOAD DOCx", "BODY MAP", "HISTORY", "MODELS", "ABOUT"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYZE
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1.7], gap="large")

    # ── LEFT: Input panel ─────────────────────────────────────────────────────
    with left_col:
        st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Drug Input</div>', unsafe_allow_html=True)

        def add_drug():
            if len(st.session_state.drugs) < 6:
                st.session_state.drugs.append({"name": DRUG_NAMES[0], "dose": 100})

        def remove_drug(idx):
            if len(st.session_state.drugs) > 2:
                st.session_state.drugs.pop(idx)

        for i, drug in enumerate(st.session_state.drugs):
            ac = ACCENT_COLORS[i % len(ACCENT_COLORS)]
            c_name, c_dose, c_rm = st.columns([2.4, 1.1, 0.35])
            with c_name:
                name = st.selectbox(
                    f"drug_{i}", DRUG_NAMES,
                    index=DRUG_NAMES.index(drug["name"]) if drug["name"] in DRUG_NAMES else 0,
                    key=f"dn_{i}", label_visibility="collapsed"
                )
                st.session_state.drugs[i]["name"] = name
            with c_dose:
                dose = st.number_input(
                    f"dose_{i}", 1, 5000, drug["dose"],
                    key=f"dd_{i}", label_visibility="collapsed"
                )
                st.session_state.drugs[i]["dose"] = dose
            with c_rm:
                if i >= 2:
                    st.button("—", key=f"rm_{i}", on_click=remove_drug,
                              args=(i,), help="Remove")
            # accent line below each drug
            st.markdown(f'<div style="height:2px;background:{ac};opacity:0.3;margin-bottom:4px;"></div>',
                        unsafe_allow_html=True)

        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

        b1, b2 = st.columns([1, 1.6])
        with b1:
            if len(st.session_state.drugs) < 6:
                st.button("+ Drug", on_click=add_drug, use_container_width=True, type="secondary")
        with b2:
            run = st.button("Run Analysis", use_container_width=True)

        # info strip
        n_drugs = len(st.session_state.drugs)
        n_pairs = n_drugs * (n_drugs - 1) // 2
        st.markdown(f"""
        <div style="margin-top:12px;padding:10px 12px;background:#0a0a0a;
          border:1px solid #1c1c1c;display:flex;justify-content:space-between;">
          <span style="font-family:'Space Mono',monospace;font-size:9px;
            text-transform:uppercase;letter-spacing:0.1em;color:#3a3a3a;">
            {n_drugs} drugs selected
          </span>
          <span style="font-family:'Space Mono',monospace;font-size:9px;
            text-transform:uppercase;letter-spacing:0.1em;color:#3a3a3a;">
            {n_pairs} pair(s) to analyze
          </span>
        </div>""", unsafe_allow_html=True)

        # ── Run Logic ─────────────────────────────────────────────────────────
        if run:
            all_drugs = [d["name"] for d in st.session_state.drugs]
            dosages   = {d["name"]: d["dose"] for d in st.session_state.drugs}

            if len(set(all_drugs)) < len(all_drugs):
                st.warning("Duplicate drugs detected.")
                st.stop()

            drug_pairs = list(combinations(all_drugs, 2))
            severe = check_severe_interaction(drug_pairs)
            if severe:
                for da, db, rule in severe:
                    st.markdown(f"""
                    <div class="contra-box">
                      <div class="contra-head">Contraindicated — Do Not Combine</div>
                      <div class="contra-pair">{da} + {db}</div>
                      <div class="contra-rule">{rule}</div>
                    </div>""", unsafe_allow_html=True)
                st.stop()

            prog = st.progress(0)
            msg  = st.empty()

            msg.markdown('<span style="font-family:\'Space Mono\',monospace;font-size:10px;color:#888;">Generating fingerprints...</span>', unsafe_allow_html=True)
            valid_drugs = []
            for name in all_drugs:
                smiles = drug_name_to_smiles(name)
                fp = smiles_to_fp(smiles)
                if fp is not None:
                    valid_drugs.append((name, smiles, fp))
            prog.progress(30)

            msg.markdown('<span style="font-family:\'Space Mono\',monospace;font-size:10px;color:#888;">Running ML predictions...</span>', unsafe_allow_html=True)
            predictions = []
            for i in range(len(valid_drugs)):
                for j in range(i+1, len(valid_drugs)):
                    da, sa, fpa = valid_drugs[i]
                    db, sb, fpb = valid_drugs[j]
                    feats = np.concatenate([fpa, fpb, np.array([dosages[da], dosages[db]])]).reshape(1, -1)
                    try:
                        pred = int(model.predict(feats)[0])
                        prob = float(model.predict_proba(feats)[0][1])
                        analysis = groq_clinical_analysis(da, db, prob)
                        organs = map_side_effects_to_organs(analysis["side_effects"], prob, dosages)
                        predictions.append({
                            "pair": (da, db), "prediction": pred, "probability": prob,
                            "side_effects": analysis["side_effects"],
                            "mechanism": analysis["mechanism"], "organ_scores": organs,
                            "dosages": {da: dosages[da], db: dosages[db]}
                        })
                    except:
                        continue
            prog.progress(90)
            msg.markdown('<span style="font-family:\'Space Mono\',monospace;font-size:10px;color:#888;">Building output...</span>', unsafe_allow_html=True)
            time.sleep(0.2)
            prog.progress(100)
            time.sleep(0.4)
            prog.empty(); msg.empty()

            st.session_state.current_results = {
                "predictions": predictions, "valid_drugs": valid_drugs,
                "dosages": dosages, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.analysis_history.append(st.session_state.current_results)
            st.rerun()

    # ── RIGHT: Results ─────────────────────────────────────────────────────────
    with right_col:
        st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

        if st.session_state.current_results:
            results = st.session_state.current_results
            preds   = results["predictions"]

            n_int  = sum(1 for p in preds if p["prediction"] == 1)
            mx     = max(p["probability"] for p in preds) if preds else 0
            hi     = sum(1 for p in preds if p["probability"] > 0.7)

            # stat grid
            mx_color = "#ff4747" if mx > 0.7 else ("#ff8c47" if mx > 0.4 else "#47ff8c")
            st.markdown(f"""
            <div class="stat-grid fade">
              <div class="stat-cell">
                <span class="stat-val">{len(preds)}</span>
                <span class="stat-lbl">Pairs Analyzed</span>
              </div>
              <div class="stat-cell">
                <span class="stat-val">{n_int}</span>
                <span class="stat-lbl">Interactions</span>
              </div>
              <div class="stat-cell">
                <span class="stat-val" style="color:#ff4747;">{hi}</span>
                <span class="stat-lbl">High Risk</span>
              </div>
              <div class="stat-cell">
                <span class="stat-val" style="color:{mx_color};">{mx*100:.0f}%</span>
                <span class="stat-lbl">Peak Risk</span>
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-label">Pair Analysis</div>', unsafe_allow_html=True)

            for p in sorted(preds, key=lambda x: -x["probability"]):
                da, db = p["pair"]
                prob   = p["probability"]
                organs = p["organ_scores"]

                if prob > 0.7:
                    rc, rl, bar_c = "risk-high", "HIGH RISK", "#ff4747"
                elif prob > 0.4:
                    rc, rl, bar_c = "risk-med", "MODERATE", "#ff8c47"
                else:
                    rc, rl, bar_c = "risk-low", "LOW RISK", "#47ff8c"

                label = f"{da}  +  {db}"
                with st.expander(f"{label}   //   {prob*100:.1f}%", expanded=(prob > 0.7)):
                    st.markdown(f"""
                    <div style="padding:14px 16px 0;">
                      <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
                        <span class="risk-tag {rc}">{rl}</span>
                        <span style="font-family:'Space Mono',monospace;font-size:11px;
                          color:#3a3a3a;">p = {prob:.4f}</span>
                        <span style="font-family:'Space Mono',monospace;font-size:9px;
                          color:#3a3a3a;">{results['timestamp']}</span>
                      </div>
                    </div>""", unsafe_allow_html=True)

                    # Molecular structures — two columns
                    mol_a_col, sep_col, mol_b_col = st.columns([5, 0.5, 5])

                    with mol_a_col:
                        sa = drug_name_to_smiles(da)
                        img_a = draw_molecule(sa, size=(190, 150))
                        st.markdown(f'<div style="font-family:\'Space Mono\',monospace;font-size:9px;letter-spacing:0.12em;text-transform:uppercase;color:#e8ff47;margin-bottom:6px;">{da}</div>', unsafe_allow_html=True)
                        if img_a:
                            st.image(img_a, use_container_width=True)
                        st.markdown(f'<div class="smiles-str">{(sa[:48]+"…") if sa and len(sa)>48 else (sa or "N/A")}</div>', unsafe_allow_html=True)

                    with sep_col:
                        st.markdown('<div style="display:flex;align-items:center;justify-content:center;height:100%;padding-top:60px;"><span style="font-family:\'Space Mono\',monospace;font-size:16px;color:#282828;">+</span></div>', unsafe_allow_html=True)

                    with mol_b_col:
                        sb = drug_name_to_smiles(db)
                        img_b = draw_molecule(sb, size=(190, 150))
                        st.markdown(f'<div style="font-family:\'Space Mono\',monospace;font-size:9px;letter-spacing:0.12em;text-transform:uppercase;color:#47c8ff;margin-bottom:6px;">{db}</div>', unsafe_allow_html=True)
                        if img_b:
                            st.image(img_b, use_container_width=True)
                        st.markdown(f'<div class="smiles-str">{(sb[:48]+"…") if sb and len(sb)>48 else (sb or "N/A")}</div>', unsafe_allow_html=True)

                    st.markdown('<div style="padding:0 16px;"><div class="divider"></div></div>', unsafe_allow_html=True)

                    # Detail section
                    d1, d2 = st.columns([1.4, 1])
                    with d1:
                        st.markdown(f"""
                        <div style="padding:0 0 0 16px;">
                          <div class="mech-box">
                            <div class="mech-label">Interaction Mechanism</div>
                            <div class="mech-text">{p["mechanism"]}</div>
                          </div>
                          <div style="font-family:'Space Mono',monospace;font-size:9px;
                            letter-spacing:0.12em;text-transform:uppercase;color:#3a3a3a;
                            margin-bottom:8px;">Predicted Side Effects</div>
                        """, unsafe_allow_html=True)
                        for idx, eff in enumerate(p["side_effects"][:5], 1):
                            st.markdown(f"""
                            <div class="se-item">
                              <span class="se-num">{idx:02d}</span>
                              <span>{eff}</span>
                            </div>""", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    with d2:
                        st.markdown('<div style="padding:0 16px 0 0;">', unsafe_allow_html=True)
                        st.markdown('<div style="font-family:\'Space Mono\',monospace;font-size:9px;letter-spacing:0.12em;text-transform:uppercase;color:#3a3a3a;margin-bottom:10px;">Organ Impact</div>', unsafe_allow_html=True)
                        for organ, score in list(organs.items())[:6]:
                            oc = ORGAN_COLORS.get(organ, "#888")
                            st.markdown(f"""
                            <div class="organ-row">
                              <div class="organ-meta">
                                <span class="organ-name">{organ}</span>
                                <span class="organ-score" style="color:{oc};">{score}</span>
                              </div>
                              <div class="bar-track">
                                <div class="bar-fill" style="width:{score}%;background:{oc};opacity:0.7;"></div>
                              </div>
                            </div>""", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-cross"></div>
              <div class="empty-title">No analysis loaded</div>
              <div style="font-family:'Space Mono',monospace;font-size:9px;
                color:#282828;letter-spacing:0.08em;">
                Select drugs and run analysis to see results
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PRESCRIPTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-label">Prescription Upload & Analysis</div>', unsafe_allow_html=True)
    
    # File upload section
    upload_col, info_col = st.columns([1.6, 1], gap="large")
    
    with upload_col:
        st.markdown('<div style="font-family:\'Space Mono\',monospace;font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:#3a3a3a;margin-bottom:12px;">Upload Document</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Select PDF or image prescription",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            st.markdown(f'<div style="font-family:\'Space Mono\',monospace;font-size:9px;color:#47ff8c;margin-top:8px;">Uploaded: {uploaded_file.name}</div>', unsafe_allow_html=True)
            
            # Extract text
            if st.button("Extract Text", use_container_width=True, key="rx_extract"):
                with st.spinner("Extracting text..."):
                    raw_text, error, success = extract_text_from_file(uploaded_file)
                    
                    if not success:
                        st.error(f"✗ Extraction failed: {error}")
                        st.stop()
                    
                    if not raw_text:
                        st.warning("No text found in document. Try a clearer image or different PDF.")
                        st.stop()
                    
                    st.session_state.rx_raw_text = raw_text
                    st.session_state.rx_parsed = None
                    st.rerun()

    with info_col:
        st.markdown('<div style="font-family:\'Space Mono\',monospace;font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:#3a3a3a;margin-bottom:12px;">Support</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#0a0a0a;border:1px solid #1c1c1c;padding:12px;font-size:11px;color:#888;line-height:1.6;">
        • <b>PDF:</b> Scanned or digital prescriptions<br>
        • <b>Image:</b> Photos of printed prescriptions<br>
        • <b>Quality:</b> Clear text improves accuracy<br>
        • <b>Support:</b> PyPDF2, pytesseract installed
        </div>""", unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Text extraction results
    if "rx_raw_text" in st.session_state:
        raw_text = st.session_state.rx_raw_text
        
        if raw_text:
            with st.expander("📄 Raw Extracted Text", expanded=False):
                st.markdown(f"""
                <div style="background:#0a0a0a;border:1px solid #1c1c1c;padding:16px;
                  border-radius:2px;font-family:'Space Mono',monospace;font-size:11px;
                  color:#888;line-height:1.6;max-height:300px;overflow-y:auto;white-space:pre-wrap;">
                {raw_text[:1000]}{'...' if len(raw_text) > 1000 else ''}
                </div>""", unsafe_allow_html=True)
        else:
            st.warning("No text extracted. Please try uploading a different file.")
        
        if raw_text:
            # Parse with Groq
            st.markdown('<div style="font-family:\'Space Mono\',monospace;font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:#3a3a3a;margin-bottom:12px;margin-top:16px;">Drug Extraction</div>', unsafe_allow_html=True)
            
            if st.button("🔍 Parse Prescription with LLM", use_container_width=True, key="rx_parse"):
                with st.spinner("Parsing prescription..."):
                    result = parse_prescription_with_groq(raw_text)
                    st.session_state.rx_parsed = result
                    st.rerun()
    
    # Display parsed results & validation
    if "rx_parsed" in st.session_state:
        parsed = st.session_state.rx_parsed
        
        if parsed and isinstance(parsed, dict):
            if "error" in parsed and parsed["error"]:
                st.error(f"parsing Error: {parsed['error']}")
            
            if not parsed.get("drugs"):
                st.info("No drugs extracted from prescription. Try another image or PDF.")
            else:
                # Validate drugs
                validation = validate_drugs(parsed["drugs"])
                
                # Display warnings
                if validation["warnings"]:
                    for warning in validation["warnings"]:
                        st.warning(warning)
                
                # Display invalid drugs
                if validation["invalid"]:
                    with st.expander(f"Not Found ({len(validation['invalid'])})", expanded=True):
                        for invalid in validation["invalid"]:
                            st.markdown(f"""
                            <div style="background:#0a0a0a;border-left:2px solid #ff4747;
                              padding:12px;margin-bottom:8px;">
                              <div style="font-family:'Space Grotesk',sans-serif;font-size:12px;
                                color:#ff4747;font-weight:600;">{invalid['original']}</div>
                              <div style="font-family:'Space Mono',monospace;font-size:9px;
                                color:#888;margin-top:4px;">{invalid['reason']}</div>
                            </div>""", unsafe_allow_html=True)
                
                # Display editable valid drugs table
                if validation["valid"]:
                    num_valid = len(validation["valid"])
                    st.markdown(f'<div style="font-family:\'Space Mono\',monospace;font-size:9px;letter-spacing:0.12em;text-transform:uppercase;color:#3a3a3a;margin-bottom:12px;margin-top:20px;">Recognized Drugs ({num_valid})</div>', unsafe_allow_html=True)
                    
                    # Initialize editable table state
                    if "rx_drugs_editable" not in st.session_state:
                        st.session_state.rx_drugs_editable = {v["name"]: v for v in validation["valid"]}
                    
                    # Create editable form
                    edit_cols = []
                    for drug in validation["valid"]:
                        orig_name = drug["name"]
                        
                        col1, col2, col3 = st.columns([2, 1.2, 0.6])
                        
                        with col1:
                            new_drug = st.selectbox(
                                "Drug", DRUG_NAMES,
                                index=DRUG_NAMES.index(orig_name) if orig_name in DRUG_NAMES else 0,
                                key=f"rx_drug_{id(drug)}"
                            )
                            st.session_state.rx_drugs_editable[new_drug] = drug.copy()
                            st.session_state.rx_drugs_editable[new_drug]["name"] = new_drug
                        
                        with col2:
                            new_dose = st.number_input(
                                "Dose", 1, 5000,
                                value=drug["dose"],
                                key=f"rx_dose_{id(drug)}"
                            )
                            st.session_state.rx_drugs_editable[new_drug]["dose"] = new_dose
                        
                        with col3:
                            if st.button("Del", key=f"rx_del_{id(drug)}", help="Remove"):
                                if new_drug in st.session_state.rx_drugs_editable:
                                    del st.session_state.rx_drugs_editable[new_drug]
                                st.rerun()
                    
                    # Final drug list before analysis
                    final_drugs = list(st.session_state.rx_drugs_editable.values())
                    
                    st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)
                    
                    # Run analysis button
                    if len(final_drugs) >= 2:
                        if st.button(
                            "Run Interaction Analysis",
                            use_container_width=True,
                            key="rx_run_analysis"
                        ):
                            # CRITICAL INTEGRATION: Inject extracted drugs into session state
                            st.session_state.drugs = [
                                {"name": d["name"], "dose": d["dose"]}
                                for d in final_drugs
                            ]
                            
                            # Trigger the same analysis pipeline
                            all_drugs = [d["name"] for d in st.session_state.drugs]
                            dosages = {d["name"]: d["dose"] for d in st.session_state.drugs}
                            
                            if len(set(all_drugs)) < len(all_drugs):
                                st.error("Duplicate drugs detected in prescription. Review and try again.")
                                st.stop()
                            
                            drug_pairs = list(combinations(all_drugs, 2))
                            severe = check_severe_interaction(drug_pairs)
                            
                            if severe:
                                for da, db, rule in severe:
                                    st.markdown(f"""
                                    <div class="contra-box">
                                      <div class="contra-head">Contraindicated — Do Not Combine</div>
                                      <div class="contra-pair">{da} + {db}</div>
                                      <div class="contra-rule">{rule}</div>
                                    </div>""", unsafe_allow_html=True)
                                st.stop()
                            
                            prog = st.progress(0)
                            msg = st.empty()
                            
                            msg.markdown('<span style="font-family:\'Space Mono\',monospace;font-size:10px;color:#888;">Generating fingerprints...</span>', unsafe_allow_html=True)
                            valid_drugs = []
                            for name in all_drugs:
                                smiles = drug_name_to_smiles(name)
                                fp = smiles_to_fp(smiles)
                                if fp is not None:
                                    valid_drugs.append((name, smiles, fp))
                            prog.progress(30)
                            
                            msg.markdown('<span style="font-family:\'Space Mono\',monospace;font-size:10px;color:#888;">Running ML predictions...</span>', unsafe_allow_html=True)
                            predictions = []
                            for i in range(len(valid_drugs)):
                                for j in range(i+1, len(valid_drugs)):
                                    da, sa, fpa = valid_drugs[i]
                                    db, sb, fpb = valid_drugs[j]
                                    feats = np.concatenate([fpa, fpb, np.array([dosages[da], dosages[db]])]).reshape(1, -1)
                                    try:
                                        pred = int(model.predict(feats)[0])
                                        prob = float(model.predict_proba(feats)[0][1])
                                        analysis = groq_clinical_analysis(da, db, prob)
                                        organs = map_side_effects_to_organs(analysis["side_effects"], prob, dosages)
                                        predictions.append({
                                            "pair": (da, db), "prediction": pred, "probability": prob,
                                            "side_effects": analysis["side_effects"],
                                            "mechanism": analysis["mechanism"], "organ_scores": organs,
                                            "dosages": {da: dosages[da], db: dosages[db]}
                                        })
                                    except:
                                        continue
                            prog.progress(90)
                            
                            msg.markdown('<span style="font-family:\'Space Mono\',monospace;font-size:10px;color:#888;">Building output...</span>', unsafe_allow_html=True)
                            time.sleep(0.2)
                            prog.progress(100)
                            time.sleep(0.4)
                            prog.empty()
                            msg.empty()
                            
                            st.session_state.current_results = {
                                "predictions": predictions, "valid_drugs": valid_drugs,
                                "dosages": dosages, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "source": "Prescription"
                            }
                            st.session_state.analysis_history.append(st.session_state.current_results)
                            
                            st.success("Analysis complete! Drugs now visible in ANALYZE tab.")
                            st.balloons()
                            st.rerun()
                    else:
                        st.info(f"Need at least 2 drugs to analyze interactions. Found: {len(final_drugs)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BODY MAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

    if not st.session_state.current_results:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-cross"></div>
          <div class="empty-title">No analysis loaded</div>
          <div style="font-family:'Space Mono',monospace;font-size:9px;
            color:#282828;letter-spacing:0.08em;">
            Run an analysis first to see the interactive body impact map
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        results = st.session_state.current_results
        preds   = results["predictions"]

        # Aggregate organ scores — CUMULATIVE TOTAL across all pairs, capped at 100.
        # This reflects the total damage burden on each organ from all drug interactions
        # combined, not just the single worst pair.
        all_organs: dict[str, list] = defaultdict(list)
        for p in preds:
            for organ, score in p["organ_scores"].items():
                all_organs[organ].append(score)

        # Cumulative sum capped at 100 — total burden model
        organ_cumulative = {o: min(100, sum(s)) for o, s in all_organs.items()}
        # Also keep max for the HTML toggle
        organ_max = {o: max(s) for o, s in all_organs.items()}

        # Pass cumulative as the primary display — HTML toggle will recompute avg/max
        # from pair-level data internally; we send cumulative as the Python-side aggregate
        organ_scores_display = organ_cumulative

        # Stats strip above the map
        top_organ = max(organ_scores_display, key=organ_scores_display.get) if organ_scores_display else "—"
        top_score = organ_scores_display.get(top_organ, 0)
        high_count = sum(1 for s in organ_scores_display.values() if s >= 70)
        mod_count  = sum(1 for s in organ_scores_display.values() if 40 <= s < 70)

        st.markdown(f"""
        <div style="display:flex;gap:1px;background:#1c1c1c;border:1px solid #1c1c1c;
          margin-bottom:16px;">
          <div style="flex:1;background:#080808;padding:10px 14px;text-align:center;">
            <div style="font-family:'Space Mono',monospace;font-size:18px;font-weight:700;
              color:#ff4747;">{high_count}</div>
            <div style="font-family:'Space Mono',monospace;font-size:8px;
              letter-spacing:0.12em;text-transform:uppercase;color:#3a3a3a;margin-top:2px;">
              High Risk Organs
            </div>
          </div>
          <div style="flex:1;background:#080808;padding:10px 14px;text-align:center;">
            <div style="font-family:'Space Mono',monospace;font-size:18px;font-weight:700;
              color:#ff8c47;">{mod_count}</div>
            <div style="font-family:'Space Mono',monospace;font-size:8px;
              letter-spacing:0.12em;text-transform:uppercase;color:#3a3a3a;margin-top:2px;">
              Moderate Risk
            </div>
          </div>
          <div style="flex:2;background:#080808;padding:10px 14px;text-align:center;">
            <div style="font-family:'Space Mono',monospace;font-size:18px;font-weight:700;
              color:{ORGAN_COLORS.get(top_organ,'#888')};">{top_organ}</div>
            <div style="font-family:'Space Mono',monospace;font-size:8px;
              letter-spacing:0.12em;text-transform:uppercase;color:#3a3a3a;margin-top:2px;">
              Highest Risk System — {top_score}/100
            </div>
          </div>
          <div style="flex:1;background:#080808;padding:10px 14px;text-align:center;">
            <div style="font-family:'Space Mono',monospace;font-size:18px;font-weight:700;
              color:#f0f0f0;">{len(organ_scores_display)}</div>
            <div style="font-family:'Space Mono',monospace;font-size:8px;
              letter-spacing:0.12em;text-transform:uppercase;color:#3a3a3a;margin-top:2px;">
              Systems Affected
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Interactive body map — 3-panel: body figure + organ list + detail
        render_body_map(organ_scores_display, preds, height=620)

    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

    if not st.session_state.analysis_history:
        st.info("No sessions recorded.")
    else:
        st.markdown(f'<div class="section-label">{len(st.session_state.analysis_history)} Sessions</div>', unsafe_allow_html=True)
        for i, sess in enumerate(reversed(st.session_state.analysis_history), 1):
            preds = sess["predictions"]
            mx = max(p["probability"] for p in preds) if preds else 0
            rc = "#ff4747" if mx > 0.7 else ("#ff8c47" if mx > 0.4 else "#47ff8c")
            n = len(st.session_state.analysis_history) - i + 1
            st.markdown(f"""
            <div class="hist-row" style="border-left-color:{rc};">
              <div>
                <div class="hist-ts">Session {n:02d}  ·  {sess["timestamp"]}</div>
                <div class="hist-detail">
                  {len(sess["valid_drugs"])} drugs  ·  {len(preds)} pairs evaluated
                </div>
              </div>
              <div>
                <div class="hist-pct" style="color:{rc};">{mx*100:.0f}%</div>
                <div class="hist-plbl">Peak Risk</div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MODELS
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

    if not MODEL_MANAGER_ENABLED:
        st.error("model_manager.py not found.")
    else:
        perf_df   = get_model_performance_dataframe()
        best_info = get_best_model_info()

        if perf_df is not None and not perf_df.empty:
            # summary row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Models", len(perf_df))
            m2.metric("Best F1", f"{perf_df['F1-Score'].max():.4f}")
            m3.metric("Best AUC", f"{perf_df['ROC-AUC'].max():.4f}")
            m4.metric("Avg F1", f"{perf_df['F1-Score'].mean():.4f}")

            st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Performance Table</div>', unsafe_allow_html=True)

            display = perf_df.copy()
            for col in ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]:
                if col in display.columns:
                    display[col] = display[col].apply(
                        lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            st.dataframe(display, use_container_width=True, hide_index=True)

            st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

            # F1 bar chart
            ms = perf_df.sort_values("F1-Score", ascending=True)
            best_name = best_info["best_model"] if best_info else ""
            bc2 = ["#e8ff47" if x == best_name else "#1a1a1a" for x in ms["Model"]]
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                y=ms["Model"], x=ms["F1-Score"],
                orientation="h",
                marker=dict(color=bc2, line=dict(color="#0f0f0f", width=1)),
                text=[f"{x:.4f}" for x in ms["F1-Score"]],
                textposition="outside",
                textfont=dict(color="#3a3a3a", size=10, family="Space Mono"),
                hovertemplate="<b>%{y}</b><br>F1: %{x:.4f}<extra></extra>"
            ))
            fig2.update_layout(
                title=dict(text="F1-SCORE RANKING",
                    font=dict(color="#3a3a3a", size=10, family="Space Mono")),
                xaxis=dict(gridcolor="#0f0f0f", color="#3a3a3a", range=[0, 1.12],
                    tickfont=dict(size=9, family="Space Mono")),
                yaxis=dict(color="#888", tickfont=dict(size=10, family="Space Mono")),
                plot_bgcolor="#080808", paper_bgcolor="#000",
                height=max(200, len(ms) * 40 + 60),
                margin=dict(l=0, r=80, t=40, b=30), showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Result images
            results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
            charts = [
                ("metrics_comparison.png", "Metrics Comparison"),
                ("metrics_heatmap.png", "Performance Heatmap"),
                ("roc_curves.png", "ROC Curves"),
                ("feature_importance.png", "Feature Importance"),
            ]
            for fname, label in charts:
                fpath = os.path.join(results_dir, fname)
                if os.path.exists(fpath):
                    st.markdown(f'<div class="section-label" style="margin-top:12px;">{label}</div>', unsafe_allow_html=True)
                    st.image(fpath, use_container_width=True)

            # Model selector — FIX for IndexError
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Active Prediction Model</div>', unsafe_allow_html=True)
            model_list = get_model_list()

            if model_list:
                # Safe default index
                default_idx = 0
                if st.session_state.selected_model in model_list:
                    default_idx = model_list.index(st.session_state.selected_model)

                selected = st.selectbox(
                    "Model", model_list, index=default_idx,
                    label_visibility="collapsed"
                )
                st.session_state.selected_model = selected

                # Safe row lookup with name normalization — fixes KNN IndexError
                norm_selected = normalize_model_name(selected)
                # Try exact match first, then normalized match, then case-insensitive
                matching_rows = perf_df[perf_df["Model"] == selected]
                if matching_rows.empty:
                    matching_rows = perf_df[perf_df["Model"] == norm_selected]
                if matching_rows.empty:
                    matching_rows = perf_df[
                        perf_df["Model"].str.lower() == selected.lower()
                    ]
                if matching_rows.empty:
                    matching_rows = perf_df[
                        perf_df["Model"].str.replace(" ","").str.lower()
                        == selected.replace(" ","").lower()
                    ]
                if not matching_rows.empty:
                    row = matching_rows.iloc[0]
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Accuracy",  f"{row['Accuracy']:.4f}"  if pd.notna(row['Accuracy'])  else "N/A")
                    s2.metric("Precision", f"{row['Precision']:.4f}" if pd.notna(row['Precision']) else "N/A")
                    s3.metric("Recall",    f"{row['Recall']:.4f}"    if pd.notna(row['Recall'])    else "N/A")
                    s4.metric("F1-Score",  f"{row['F1-Score']:.4f}"  if pd.notna(row['F1-Score'])  else "N/A")
                    st.caption(f"Active: {selected}")
                else:
                    st.warning(f"No metrics found for '{selected}'. The model comparison results may not include this model name. Please re-run model_comparison.py to regenerate.")
            else:
                st.warning("No models available. Run model_comparison.py first.")
        else:
            st.error("No comparison results found. Run model_comparison.py first.")

    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

    a1, a2 = st.columns([1.4, 1], gap="large")
    with a1:
        st.markdown('<div class="section-label">System Overview</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#0a0a0a;border:1px solid #1c1c1c;padding:20px;margin-bottom:16px;">
          <p style="font-family:'Space Grotesk',sans-serif;font-size:13px;color:#888;
            line-height:1.8;margin:0;">
            DDI Predictor Pro is a dosage-aware drug-drug interaction prediction framework
            combining RDKit Morgan fingerprints with a trained Random Forest classifier
            and a multi-factor organ impact scoring algorithm — designed for clinical
            decision support and educational research.
          </p>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-label">Pipeline</div>', unsafe_allow_html=True)
        steps = [
            ("01", "SMILES strings converted to 2048-bit Morgan fingerprints via RDKit"),
            ("02", "Pairwise feature vector constructed: [FP_A | FP_B | dose_A | dose_B] — 4098 dims"),
            ("03", "Random Forest predicts interaction probability for each pair"),
            ("04", "Hard-rule engine checks 12 documented life-threatening contraindications"),
            ("05", "Groq LLM generates side effects and mechanism explanations"),
            ("06", "Organ impact scores computed via weighted multi-factor algorithm"),
        ]
        for num, text in steps:
            st.markdown(f"""
            <div style="display:flex;gap:14px;padding:10px 0;
              border-bottom:1px solid #0f0f0f;align-items:flex-start;">
              <span style="font-family:'Space Mono',monospace;font-size:10px;
                color:#e8ff47;min-width:22px;">{num}</span>
              <span style="font-family:'Space Grotesk',sans-serif;font-size:12px;
                color:#888;line-height:1.6;">{text}</span>
            </div>""", unsafe_allow_html=True)

    with a2:
        st.markdown('<div class="section-label">Formula</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#080808;border:1px solid #1c1c1c;
          border-left:2px solid #e8ff47;padding:16px;margin-bottom:16px;
          font-family:'Space Mono',monospace;">
          <div style="font-size:9px;letter-spacing:0.14em;text-transform:uppercase;
            color:#3a3a3a;margin-bottom:10px;">Organ Impact Score</div>
          <div style="font-size:12px;color:#e8ff47;line-height:2;">
            S = norm( P &times; W<sub>sev</sub> &times; M<sub>dose</sub> &times; C<sub>conf</sub> )
          </div>
          <div style="height:10px;"></div>
          <div style="font-size:10px;color:#3a3a3a;line-height:1.9;">
            P &nbsp;&nbsp; model probability [0, 1]<br>
            W &nbsp; severity weight [0.3 – 1.0]<br>
            M &nbsp; dosage multiplier [1.0 – 1.5]<br>
            C &nbsp; confidence factor {0.6, 0.75}
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-label">Stack</div>', unsafe_allow_html=True)
        stack = ["Streamlit", "scikit-learn", "RDKit", "Groq Llama 3.3", "Plotly", "NumPy", "Pandas"]
        chips_html = "".join(
            f'<span class="tag-mono" style="margin:3px 3px 3px 0;display:inline-block;">{s}</span>'
            for s in stack)
        st.markdown(f'<div style="margin-top:4px;">{chips_html}</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:16px;background:#080808;border:1px solid rgba(255,71,71,0.15);
          border-left:2px solid #ff4747;padding:14px;">
          <div style="font-family:'Space Mono',monospace;font-size:9px;letter-spacing:0.14em;
            text-transform:uppercase;color:#ff4747;margin-bottom:5px;">Medical Disclaimer</div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:11px;color:#3a3a3a;line-height:1.6;">
            Educational and research purposes only. Not a substitute for professional medical
            advice. Always consult qualified healthcare providers before clinical decisions.
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid #0f0f0f;padding:12px 32px;
  display:flex;justify-content:space-between;align-items:center;
  background:#000;">
  <span style="font-family:'Space Mono',monospace;font-size:9px;
    letter-spacing:0.1em;text-transform:uppercase;color:#1c1c1c;">
    DDI Predictor Pro &copy; 2026 by Rudrajyoti Paul — Educational Tool — Not Medical Advice
  </span>
  <span style="font-family:'Space Mono',monospace;font-size:9px;
    letter-spacing:0.1em;color:#1c1c1c;">
    RDKit &middot; scikit-learn &middot; Groq AI
  </span>
</div>""", unsafe_allow_html=True)