# Prescription Analysis - Quick Start

## What's New?

A new **RX ANALYSIS** tab allows you to upload prescriptions (PDF or photo) and automatically extract drug information for interaction analysis.

## Installation (First Time)

```bash
# Install required packages
pip install PyPDF2 pytesseract pillow

# Install Tesseract OCR (for image support)
# macOS:
brew install tesseract

# Linux:
sudo apt-get install tesseract-ocr

# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

## Usage (3 Steps)

### Step 1: Upload Prescription
1. Click **RX ANALYSIS** tab
2. Use file uploader to select:
   - **PDF:** Digital or scanned prescription
   - **Image:** Photo of printed prescription (JPG, PNG, TIFF, BMP)
3. Click **Extract Text** button
4. Review extracted text in the expander

### Step 2: Parse Drugs
1. Click **Parse Prescription with LLM** button
2. Wait for Groq to extract structured drug data
3. Review warnings (if any):
   - ❌ Not Found: Drug not in system
   - ⚠️ Warnings: Dose out of range, text clarity issues

### Step 3: Review & Run
1. Edit extracted drugs:
   - **Drug name:** Dropdown to select from system
   - **Dose:** Input 1-5000 mg
   - **Delete:** Remove unwanted drugs
2. Click **▶ Run Interaction Analysis**
3. Wait for analysis (8-16 seconds)
4. View results in:
   - **BODY MAP** tab: Organ impact visualization
   - **HISTORY** tab: Session record

## Example

```
Scanned Prescription:
  Aspirin 500mg 3x daily
  Ibuprofen 400mg 2x daily

After upload & parsing:
  ✓ Aspirin — 500 mg (matched)
  ✓ Ibuprofen — 400 mg (matched)

Results:
  - Risk: 75% (High)
  - Top organ affected: Blood/Coagulation
  - Interaction: "Increased bleeding risk"
```

## Supported Formats

| Format | Support | Quality |
|--------|---------|---------|
| PDF (text-based) | ✓ Best | Excellent |
| PDF (scanned) | ✓ Good | Good |
| Image (300+ dpi) | ✓ Good | Good |
| JPG/PNG/TIFF/BMP | ✓ All | Depends on quality |
| Unclear images | ✓ Works | Limited |

## Limitations

- **Drug Database:** Limited to ~15 common drugs (see list in ABOUT tab)
- **Unrecognized Drugs:** Can be edited in review table
- **Dose Extraction:** Simple numeric parsing (works for "500 mg", "2x daily")
- **Language:** English prescription text only
- **API:** Requires active Groq API key

## Troubleshooting

### No text extracted from PDF?
- PDF may be image-based → Use photo/scan instead
- Try uploading as image (convert PDF to JPG)

### Drug marked as "Not Found"?
- Drug not in system database
- Select closest match from dropdown
- Or contact system admin to add drug

### Tesseract error on Windows?
- Download and install Tesseract from:
  https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH or configure in code

### Results don't match manual input?
- Check edited doses in review table
- Ensure capitalization matches (e.g., "Aspirin" not "aspirin")
- Compare risk % in BODY MAP and HISTORY tabs

## Tips & Best Practices

✅ **Use clear images:** Better OCR accuracy  
✅ **Check extracted drugs:** Review before clicking "Run"  
✅ **Verify dosages:** Ensure numeric dose is correct  
✅ **Validate results:** Compare with manual entry if unsure  

❌ **Don't:** Rely on incomplete prescriptions (need ≥2 drugs)  
❌ **Don't:** Upload non-prescription documents  
❌ **Don't:** Expect perfect OCR on low-quality images  

## Technical Details

- **Text Extraction:** PyPDF2 (PDF) + Tesseract (image)
- **Drug Parsing:** Groq LLM (llama-3.3-70b-versatile)
- **Validation:** Fuzzy matching against DRUG_NAMES
- **Analysis:** Identical pipeline to manual mode

## Next Steps

After running prescription analysis:

1. **View Results:**
   - BODY MAP tab → Interactive organ visualization
   - HISTORY tab → Session summary

2. **Compare Interactions:**
   - Pair cards show risk %
   - Expand to see mechanism
   - Review organ impact scores

3. **Export & Share:**
   - Screenshots of body map
   - History entries timestamped
   - Share with healthcare providers

---

## Contact Support

If you encounter issues:
1. Check Python/dependency versions
2. Verify Groq API key is set
3. Review `PRESCRIPTION_ANALYSIS_GUIDE.md` (technical docs)
4. Check system logs for error messages
