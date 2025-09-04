# Motion2Meaning: Contestable AI for Parkinsonian Gait Interpretation

Motion2Meaning is a **clinician-centered framework** for contestable interpretation of Parkinson’s Disease (PD) gait data.  
It integrates **wearable sensor analysis, explainable AI (XAI), and contestable system design** into a single workflow that prioritizes transparency, accountability, and human oversight.

---

## Key Features
- **Gait Data Visualization Interface (GDVI)**  
  Interactive web-based tool to explore raw vertical Ground Reaction Force (vGRF) signals with stride, stance, and swing markers.

- **1D-CNN Diagnostic Pipeline**  
  End-to-end prediction of **Hoehn & Yahr severity scores** from raw gait signals.

- **Cross-Modal Explanation Discrepancy (XMED)**  
  Compares Grad-CAM and LRP explanations to detect inconsistent or unreliable model predictions.

- **Contestable Interpretation Interface (CII)**  
  A dashboard for clinicians to review, challenge, and override AI outputs.  
  - Structured **“Contest & Justify” workflow**  
  - LLM-powered justifications grounded in clinical evidence  
  - Immutable logging of disagreements and resolutions  

---

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/motion2meaning.git
   cd motion2meaning
   ```
2. Install dependencies:
```pip install -r requirements.txt```

3. Run the web-based dashboard:
   ```python app.py```
