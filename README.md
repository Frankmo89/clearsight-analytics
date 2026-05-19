

https://github.com/user-attachments/assets/cace6419-dd8c-4262-8358-8a1fd07e2b9c

# 👁️ ClearSight Analytics

> **Multimodal Clinical Decision Support System (CDSS) for Diabetic Care Management**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-red)](https://xgboost.readthedocs.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-BioBERT-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-BioBERT-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/whoukcode/finalcapstone)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.1-f59e0b)](https://groq.com/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)](LICENSE)

**ClearSight Analytics** is a high-fidelity multimodal AI platform designed to assist healthcare professionals in high-stakes clinical decision-making. It combines traditional machine learning, deep learning, computer vision, NLP, and generative AI to provide a 360-degree risk assessment of diabetic patients — from hospital readmission probability to retinal complication screening and AI-powered drug recommendations.

> ⚠️ **This is an academic Capstone Project.** It is not a certified medical device and must not be used for real clinical diagnosis.

---
> 🙋‍♂️ **Recruiters & Hiring Managers:** This was a collaborative team project. If you are evaluating my individual technical skills, please see my **[PORTFOLIO.md](./PORTFOLIO.md)** for a deep dive into my specific contributions (Consensus Architecture, Model 5, and the Streamlit UI/UX).
---

## ✨ Key Features

- **6 Specialized AI Models** running across a single unified inference pipeline
- **BioBERT (LoRA) NLP** — 110M-parameter biomedical transformer classifying drug effectiveness from patient reviews with 90% accuracy
- **Drug Effectiveness Recommender** — innovation engine surfacing ranked alternative medications based on 192,482 patient-reported outcomes
- **Grad-CAM Explainability** — visual heatmaps highlighting pathological retinal regions
- **AI Clinical Copilot** — context-aware LLM (Llama 3.1) that injects live patient metrics, model outputs, and drug recommendations into every prompt
- **Quick Load Presets** — 5 pre-built clinical scenarios for rapid demo and testing
- **EHR Export** — one-click Markdown clinical summary including drug recommendation output
- **Clinical Safety Guardrails** — mandatory disclaimers, `temperature=0.2` lock, and OOD warnings on all AI outputs
- **Zero-setup model loading** — all model weights auto-downloaded from HuggingFace on first run

---

## 🏛️ System Architecture

The platform operates on a **Consensus Inference Engine**, integrating six specialized models:

### Tier 1 — Predictive Risk (M1 & M2)
| Model | Type | Task |
|-------|------|------|
| **M1** | XGBoost Ensemble | 30-day readmission risk from structured EHR data |
| **M2** | Keras DNN | Parallel readmission probability for consensus scoring |

### Tier 2 — Computer Vision (M3)
| Model | Type | Task |
|-------|------|------|
| **M3** | ResNet50 CNN | Diabetic Retinopathy screening from fundus photographs |
| **Grad-CAM** | Explainable AI | Activation heatmaps over hemorrhages, exudates, and microaneurysms |

### Tier 3 — NLP, Operations & Innovation (M4, M5, M6)
| Model | Type | Task |
|-------|------|------|
| **M4** | BioBERT (LoRA) + Drug/Condition Embeddings | Clinical note sentiment + medication effectiveness classification |
| **M5** | Capacity Classifier | Length of Stay (LOS) prediction for bed management |
| **M6** | Drug Effectiveness Recommender | Ranks alternative drugs by patient-reported outcomes per condition |

### Tier 4 — Generative AI Copilot
Powered by **Llama 3.1-8b-instant via Groq API** for sub-second latency. Dynamically injects live patient metrics, all model outputs, and drug recommendation rankings into the LLM context to generate personalized discharge protocols, risk driver rankings, and clinical syntheses.

---

## 🧠 Model 4 — BioBERT (LoRA) NLP Classifier

The NLP model is the most technically sophisticated component of the system.

- **Base model:** `dmis-lab/biobert-base-cased-v1.2` — 110M parameters pretrained on PubMed abstracts and PMC full-text biomedical literature
- **Fine-tuning:** LoRA (Low-Rank Adaptation) applied to Q and V attention projections — only **0.87% of parameters train** (~948k of 110M), preventing overfitting
- **Metadata conditioning:** Drug and condition embeddings (32-dim each) allow the same review text to be interpreted differently depending on which medication is being discussed
- **Training data:** 192,482 patient medication reviews across 3,570 drugs and 906 conditions
- **Performance:** Weighted F1 = **0.9018**, Accuracy = **90%**
- **Classes:** Highly Effective / Somewhat Effective / Ineffective

All model weights are hosted on [HuggingFace](https://huggingface.co/whoukcode/finalcapstone) and downloaded automatically on first run — no manual setup required.

---

## 💊 Model 6 — Drug Effectiveness Recommender (Innovation)

Built on top of M4's predictions, this innovation engine pre-computes drug effectiveness rankings from 192,482 patient reviews and surfaces ranked alternatives whenever a drug classification is made.

- Requires a minimum of **20 patient reviews** per drug/condition combination for inclusion
- Ranking uses a weighted effectiveness score: `Highly Effective × 1.0 + Somewhat Effective × 0.5`
- Always displayed in the webapp — heading adapts to the M4 result:
  - *"Metformin was Ineffective for Diabetes, Type 2 — Recommended Alternatives Likely to Produce Better Outcomes"*
  - *"Metformin was Highly Effective — Other Drugs That Have Also Shown Strong Results"*
- Rankings CSV hosted on HuggingFace and auto-downloaded by the app on startup
- Drug recommendations are injected into the AI Copilot context and clinical report export

---

## 🛡️ Clinical Safety & Ethics

| Guardrail | Implementation |
|-----------|---------------|
| **Hallucination control** | All generative outputs locked at `temperature: 0.2` |
| **Mandatory disclaimers** | Every AI response enforced with a `CRITICAL RULE` system instruction |
| **OOD detection UX** | Explicit warning on retinal uploader for non-fundus images |
| **No definitive diagnosis** | CDSS system prompt explicitly prohibits diagnostic conclusions |
| **Audit trail** | All prediction events logged via Python `logging` module |

---

## 📂 Project Structure

```
final-capstone-clearsight-analytics/
├── webapp/
│   └── app.py                    # Main Streamlit application
├── models/
│   ├── model1_traditional_ml/    # XGBoost readmission ensemble
│   ├── model2_deep_learning/     # Keras DNN readmission model
│   ├── model3_cnn/               # ResNet50 retinal screening + Grad-CAM
│   ├── model4_nlp_classification/# BioBERT (LoRA) clinical NLP classifier
│   ├── model5_innovation/        # LOS / capacity planning classifier
│   └── model6_innovation/        # Drug effectiveness recommender
├── pipelines/                    # Data cleaning & feature engineering
├── notebooks/                    # EDA notebooks per team member
├── data/processed/               # Processed datasets (gitignored)
├── test_data/                    # Model prediction outputs
├── output_templates/             # Per-model result CSV templates
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Deployment

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com/) (for the AI Copilot)

### Setup

```bash
# 1. Clone
git clone https://github.com/Frankmo89/clearsight-analytics.git
cd clearsight-analytics

# 2. Create and activate virtual environment
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

### Secrets Configuration

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

### Run

```bash
streamlit run webapp/app.py
```

Navigate to `http://localhost:8501` in your browser.

> **Note:** On first run, the app will automatically download all model weights and the drug rankings CSV from HuggingFace (`whoukcode/finalcapstone`). The BioBERT model is ~435 MB — allow 1–2 minutes for the initial download.

---

## 🛠️ Technical Stack

| Layer | Technology |
|-------|-----------|
| UI Framework | Streamlit 1.32+ with custom glassmorphism CSS |
| ML — Tabular | XGBoost, scikit-learn, joblib |
| ML — Deep Learning | TensorFlow / Keras 2.15 |
| ML — Computer Vision | PyTorch, torchvision (ResNet50), OpenCV |
| ML — NLP | PyTorch, HuggingFace Transformers, BioBERT (LoRA) |
| Model Hosting | HuggingFace Hub (auto-download on first run) |
| Generative AI | Llama 3.1 via Groq API (OpenAI-compatible client) |
| Data Processing | pandas, NumPy, scikit-learn pipelines |
| Explainability | Grad-CAM (manual implementation, no third-party lib) |

---

## 👥 Team

| Role | Name | GitHub |
|------|------|--------|
| **ML / DNN Lead** | Francisco Molina | [@Frankmo89](https://github.com/Frankmo89) |
| **NLP Lead** | Wesley Houk | [@wesleyhouk](https://github.com/wesleyhouk) |
| **CNN / CV Lead** | Doug Bacon | [@Cooley5632](https://github.com/@Cooley5632)|
| **Data Engineering** | Francisco Molina | [@Frankmo89](https://github.com/Frankmo89) |

---

*⚠️ **Disclaimer:** This software is an academic Capstone Project for demonstration purposes only. It is not a certified medical device and should not be used for actual clinical diagnosis.*
