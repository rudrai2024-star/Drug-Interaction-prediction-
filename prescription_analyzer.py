"""
Prescription Analysis Module for DDI Predictor Pro

Provides functions for extracting text from prescriptions (PDF/image) and parsing
drug information using Groq LLM, then validating against the system drug database.

This module integrates seamlessly with the existing DDI prediction pipeline.
"""

import json
import re
from typing import Tuple, Dict, List, Any


def extract_text_from_file_standalone(uploaded_file) -> Tuple[str, str, bool]:
    """
    Standalone text extraction from PDF or image file.
    Requires PyPDF2 and pytesseract to be installed.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        (text, error_msg, success_flag)
    """
    if uploaded_file is None:
        return "", "No file uploaded", False

    try:
        if uploaded_file.type == "application/pdf":
            try:
                import PyPDF2
            except ImportError:
                return "", "PyPDF2 not installed. Install with: pip install PyPDF2", False
            
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text.strip(), None, bool(text.strip())

        elif uploaded_file.type in ["image/png", "image/jpeg", "image/jpg", "image/tiff", "image/bmp"]:
            try:
                import pytesseract
                from PIL import Image
                import io
            except ImportError:
                return "", "pytesseract/Pillow not installed. Install with: pip install pytesseract pillow", False
            
            image = Image.open(io.BytesIO(uploaded_file.getvalue()))
            text = pytesseract.image_to_string(image)
            return text.strip(), None, bool(text.strip())

        else:
            return "", f"Unsupported file type: {uploaded_file.type}. Use PDF or image (PNG, JPG, TIFF, BMP).", False

    except Exception as e:
        return "", f"Error extracting text: {str(e)}", False


def parse_prescription_groq_protocol(prescription_text: str, groq_client=None, model: str = "llama-3.3-70b-versatile") -> Dict[str, Any]:
    """
    Use Groq LLM to parse prescription text and extract structured drug data.
    
    Args:
        prescription_text: Raw prescription text from PDF/image
        groq_client: Groq API client instance
        model: Model to use (default: llama-3.3-70b-versatile)
    
    Returns:
        {
            "drugs": [{"name": str, "dose": str, "frequency": str}, ...],
            "raw_text": str (first 500 chars),
            "confidence": float (0-1),
            "error": str (if any)
        }
    """
    if groq_client is None:
        return {
            "drugs": [],
            "raw_text": prescription_text[:500] if prescription_text else "",
            "confidence": 0.0,
            "error": "Groq client not initialized"
        }

    if not prescription_text or len(prescription_text) < 10:
        return {
            "drugs": [],
            "raw_text": prescription_text[:500] if prescription_text else "",
            "confidence": 0.0,
            "error": "Prescription text too short or empty"
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
            model=model,
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

        result["raw_text"] = prescription_text[:500] if prescription_text else ""
        result["confidence"] = result.get("confidence", 0.5)

        return result

    except json.JSONDecodeError as e:
        return {
            "drugs": [],
            "raw_text": prescription_text[:500] if prescription_text else "",
            "confidence": 0.0,
            "error": f"Failed to parse response JSON: {str(e)}"
        }
    except Exception as e:
        return {
            "drugs": [],
            "raw_text": prescription_text[:500] if prescription_text else "",
            "confidence": 0.0,
            "error": f"Groq API error: {str(e)}"
        }


def validate_drugs_against_db(drugs_list: List[Dict[str, str]], available_drugs: List[str]) -> Dict[str, Any]:
    """
    Validate extracted drugs against system drug list.
    
    Args:
        drugs_list: List of extracted drugs [{"name": str, "dose": str, "frequency": str}, ...]
        available_drugs: List of valid drug names in the system (e.g., DRUG_NAMES from app)
    
    Returns:
        {
            "valid": [{"name": str, "dose": int, "matched": bool, ...}, ...],
            "invalid": [{"original": str, "reason": str}, ...],
            "warnings": [str, ...]
        }
    """
    valid = []
    invalid = []
    warnings = []

    available_drugs_lower = {d.lower(): d for d in available_drugs}

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
                numbers = re.findall(r'\d+', dose_str)
                if numbers:
                    dose_num = int(numbers[0])
                    if dose_num < 1 or dose_num > 5000:
                        dose_num = 100
                        warnings.append(
                            f"{matched_name}: Dose {drug.get('dose')} outside typical range, using default 100 mg"
                        )
            except:
                warnings.append(
                    f"{matched_name}: Could not parse dose '{dose_str}', using default 100 mg"
                )

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
            warnings.append(
                f"Drug '{name}' not recognized. Available: {', '.join(available_drugs[:5])}..."
            )

    return {
        "valid": valid,
        "invalid": invalid,
        "warnings": warnings
    }


# Export functions
__all__ = [
    "extract_text_from_file_standalone",
    "parse_prescription_groq_protocol",
    "validate_drugs_against_db",
]
