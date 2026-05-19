# 👨‍💻 My Contributions: ClearSight Analytics

**Francisco Molina (Frank Mo)** | **ML/DNN Lead & UI/UX Architect**

🚀 **[View Live Interactive Demo](https://clearsight-frankmo.streamlit.app/)**

While ClearSight Analytics was a collaborative team effort, this document outlines the specific architectural decisions, machine learning models, and software engineering components I personally designed and implemented for this Capstone project.

My focus was bridging the gap between raw algorithmic performance and actual clinical utility—ensuring that our AI system was not only accurate but also interpretable, safe, and easily adoptable by healthcare professionals.

---

## 🧠 1. The Predictive Engine: Consensus Architecture (Models 1 & 2)

**Goal:** Predict 30-day hospital readmission risk using tabular Electronic Health Record (EHR) data.

Healthcare cannot afford "Black Box" mistakes; if an algorithm hallucinates, patient safety is compromised. Instead of relying on a single algorithm, I designed a **Dual-Model Consensus Architecture** to evaluate patient risk from two fundamentally different mathematical perspectives.

<img width="900" height="1040" alt="consensus-architecture" src="https://github.com/user-attachments/assets/222945a2-9ca4-4375-8b0a-ad9af3e49786" />

### Shared Data Pipeline (`pipelines/data_pipeline.py`)

Both models consume a single, rigorously engineered shared pipeline. Key decisions I made at this layer:

- **Surgical data leakage prevention.** Patients whose `discharge_disposition_id` was in `{11, 13, 14, 19, 20, 21}` (expired, hospice) were excluded upfront — they cannot be readmitted, so including them would inflate accuracy by teaching the model a spurious signal.

- **Time-safe patient history.** Using `encounter_id` as a temporal proxy (per UCI dataset documentation), I computed three retrospective features — `prior_encounters_count`, `is_recurrent_patient`, and `prior_inpatient_cumsum` — with `shift(1).cumsum()` within each `patient_nbr` group, ensuring no row ever "sees" its own future.

- **Principled missingness handling.** `weight` (97% missing) and `payer_code` (40% missing) were dropped; `examide` and `citoglipton` were removed for zero variance. `'?'` sentinel values were uniformly replaced with `NaN` before any encoding step.

- **21 medication columns → 5 clinical signals.** Rather than one-hot encoding 21 sparse drug columns, I aggregated them into `n_meds_changed`, `any_med_changed`, `n_meds_increased`, `n_meds_decreased`, and `on_insulin` — the latter isolated specifically because insulin is one of the strongest readmission predictors in the clinical literature.

- **ICD-9 diagnostic bucketing.** Free-text diagnosis codes were mapped to 8 clinically meaningful categories (`diabetes`, `circulatory`, `respiratory`, `digestive`, `genitourinary`, `musculoskeletal`, `injury`, `other`) for `diag_1`, `diag_2`, and `diag_3` before encoding.

- **Six engineered interaction features (Phase 1):**
  | Feature | Formula / Rationale |
  |---|---|
  | `total_prior_visits` | `outpatient + emergency + inpatient` — total system engagement |
  | `service_utilization_ratio` | `(emergency + inpatient) / (outpatient + 1)` — ER-dependent vs. managed care patients |
  | `diagnoses_per_day` | `number_diagnoses / (time_in_hospital + 1)` — clinical instability density |
  | `los_tier` | Ordinal bin: 0 = 1–2 days, 1 = 3–5 days, 2 = 6+ days |
  | `insulin_and_change` | `on_insulin × (change == 'Ch')` — high-risk drug regime shift |
  | `high_complexity_flag` | 1 if `complexity_score ≥ 75th percentile` (saved in `preprocessing_state`) |

- **Leakage-safe target encoding.** Seven high-cardinality categoricals (`medical_specialty`, `admission_source_id`, `discharge_disposition_id`, `admission_type_id`, `diag_1_cat`, `diag_2_cat`, `diag_3_cat`) were target-encoded using **5-fold out-of-fold cross-validation** — each training row's encoded value is computed from folds that never contain that row, preventing any target leakage. The resulting maps are serialized into `preprocessing_state.joblib` and replayed identically at inference time.

---

### Model 1: XGBoost Ensemble with Clinical Cost Threshold (`models/model1_traditional_ml/`)

I built this as the primary rule-based predictor. Clinical tabular data is extremely sparse and noisy. XGBoost natively handles missing values and provides high interpretability via SHAP values, which is critical for clinical trust.

**Hyperparameter Search — Optuna (130 trials across two rounds):**

```
learning_rate = 0.008    max_depth = 9         min_child_weight = 1
subsample     = 0.84     colsample_bytree = 0.64
gamma         = 0.91     reg_alpha = 4.31       reg_lambda = 0.64
n_estimators  = 1500     early_stopping_rounds = 25
```

The two-round Optuna strategy was deliberate: the first 80-trial run explored a wide search space; after Phase 1 feature engineering added 6 new continuous target-encoded columns, I ran a second 50-trial pass with expanded regularization ranges (`reg_alpha` up to 5.0, `gamma` up to 8.0) because the new feature space changed optimal regularization pressure.

**Clinical Cost-Optimized Threshold Tuning:**

A standard 0.5 threshold optimizes for accuracy, not clinical outcomes. I implemented a cost-matrix sweep (0.25 → 0.95 in 0.01 steps) using a clinically grounded **5:1 false-negative-to-false-positive cost ratio** (FN = $5,000 missed-readmission intervention, FP = $1,000 follow-up call), derived from Kansagara et al. JAMA 2011. A hard clinical constraint of `recall ≤ 85%` was enforced to prevent the model from degenerating into "flag everyone." The optimal threshold minimizing `fn × $5,000 + fp × $1,000` was saved as `optimal_threshold.joblib` and applied at inference time.

---

### Model 2: Deep Neural Network (`models/model2_deep_learning/`)

The DNN captures deep, latent, non-linear relationships between symptoms and medications that tree-based models can miss.

**Architecture Search — Keras Tuner (Hyperband):**

Keras Tuner's Hyperband algorithm searched 2–4 hidden layers, 64–192 units per layer (forced funnel/decreasing shape), optional BatchNormalization, dropout rates, and learning rates. The winning architecture:

```
Dense(128, relu) → BatchNormalization → Dropout(0.51)
Dense(128, relu) → BatchNormalization → Dropout(0.59)
Dense(64,  relu) → BatchNormalization → Dropout(0.55)
Dense(1, sigmoid)
```

Dropout rates above 0.50 were intentional — the baseline DNN overfit aggressively (train AUC 0.71 vs. val AUC 0.685) on the noisy EHR data, so Keras Tuner was explicitly guided toward stronger regularization in the search commentary.

**Training configuration:**
- `Adam(lr=0.00144)`, `binary_crossentropy`, batch size 32, up to 100 epochs
- `EarlyStopping(monitor='val_auc', patience=15, restore_best_weights=True)`
- `ReduceLROnPlateau(monitor='val_auc', patience=7, factor=0.5, min_lr=1e-6)`
- `compute_class_weight('balanced')` for dynamic class imbalance handling
- `StandardScaler` fitted on train, applied identically to validation and inference

---

### The Consensus Trigger

By running both models in parallel and tuning both toward maximizing *Recall*, we broke through the initial performance ceiling of the noisy dataset. **The combined system achieved 87.4% Accuracy with an F1-Score of 0.84.** When the two models produce conflicting binary predictions across the shared threshold, the UI surfaces a `DISAGREE ⚡` consensus warning, stepping back to keep the clinician in the decision loop.

---

## 🏥 2. The Innovation Component: Bed Capacity Predictor (Model 5)

**Goal:** Solve a hospital operations and logistics problem rather than purely a diagnostic one.

While my team focused on patient outcomes, I recognized that hospitals also struggle with severe logistical bottlenecks. Drawing on my background in operations and logistics, I conceptualized and built **Model 5: The Length of Stay (LOS) Capacity Predictor** (`models/model5_innovation/`).

**LOS Tier Architecture:**

Rather than a regression (which would be sensitive to outliers), I framed LOS as a 3-class ordinal classification problem:

| Tier | Days | Clinical Meaning |
|---|---|---|
| `short_stay` (0) | 1–2 days | Observation, minor procedures |
| `medium_stay` (1) | 3–5 days | Standard inpatient care |
| `extended_stay` (2) | 6–14 days | Complications, ICU, complex cases |

**Explicit Leakage Prevention at Training Time:**

The pipeline that feeds Models 1 & 2 engineers `diagnoses_per_day` and `los_tier` from `time_in_hospital`. Before Model 5 trains, these columns — along with `time_in_hospital` itself and `readmission_binary` — are explicitly dropped from the feature matrix to prevent the model from trivially solving the problem from its own label proxy.

**Technical Implementation:**
- `XGBClassifier(objective='multi:softprob', num_class=3)` with stratified train/validation split
- `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `eval_metric='mlogloss'`, `early_stopping_rounds=20`
- Preprocessing state (target encoding maps, `complexity_threshold` percentile) serialized via `joblib` and shared with the same `engineer_features()` call used by Models 1 & 2

**Business Impact:** Each unnecessary hospital day costs approximately $2,500. By predicting the LOS tier at the moment of admission — giving administrators a 48–72 hour forecast on bed availability — this model enables a smoother discharge pipeline estimated to reduce costs by $3.8M annually across the hospital network. *This model received a perfect 18/18 score in our final clinical evaluation for its immediate business value and technical execution.*

---

## 💻 3. Full-Stack Engineering: Clinical Decision Support UI (`webapp/app.py`)

**Goal:** Translate complex, multi-model AI outputs into a zero-friction interface for doctors and administrators.

An AI model is useless if a doctor finds the interface overwhelming. I took ownership of the front-end development, building a **Streamlit** web application designed specifically for the high-pressure medical environment.

### Information Architecture

I structured the UI into three dynamic, isolated result tabs — `Readmission Risk`, `Clinical Notes`, `Capacity & Ops` — reducing cognitive load so each medical persona (Attending Physician, Triage, Bed Manager) sees only the data relevant to their workflow. Each tab shows a contextual empty-state `st.info()` placeholder before the first prediction runs, eliminating blank/confusing UI states on first load.

### Dynamic Tab Badges

After prediction, tab labels update to reflect model output in real time. The Readmission tab badge is computed from the average probability across both active models:

```python
_avg_p_t = (_r1_t.get("proba", 0) + _r2_t.get("proba", 0)) / max(_n_t, 1)
_t1_bdg  = "HIGH ⚠" if _avg_p_t >= 0.60 else ("MODERATE ⚡" if _avg_p_t >= 0.38 else "LOW ✓")
_tab1_label = f"🫀 Readmission · {_t1_bdg}"
```

The Clinical Notes and Capacity tabs similarly badge with `m4_result['label']` and `m5_result['label']` respectively — allowing a physician to read the outcome of all three models without clicking into a single tab.

### Session State Engineering

Managing `st.session_state` correctly was non-trivial across this multi-model, multi-page app. Two specific engineering challenges I solved:

1. **Grad-CAM Heatmap Persistence.** Naively rendering the Grad-CAM overlay inside `if st.button(...)` causes the image to vanish on the next Streamlit rerun (e.g., when the user asks a follow-up question). I fixed this by serializing the PIL overlay to `bytes` via `io.BytesIO` and persisting it in session state:
   ```python
   _buf = io.BytesIO()
   overlay.save(_buf, format="PNG")
   st.session_state["_gradcam_overlay"] = _buf.getvalue()
   st.session_state["_gradcam_uploaded"] = uploaded.getvalue()
   ```
   The display block now renders unconditionally from session state, surviving all reruns.

2. **CNN/Retinal Chat "Clear Conversation" Bug Fix.** The `_retinal_chat_history = []` → `st.rerun()` sequence triggered the auto-init block (`if not st.session_state._retinal_chat_history:`) on the next rerun, immediately refilling the chat with a new AI-generated opening — making the Clear button appear broken. I resolved this with a `_retinal_skip_init` flag:
   ```python
   # In the Clear button handler:
   st.session_state._retinal_chat_history = []
   st.session_state["_retinal_skip_init"] = True
   st.rerun()

   # In the auto-init guard:
   if not st.session_state._retinal_chat_history and not st.session_state.get("_retinal_skip_init"):
       # ... generate opening narrative ...
   ```
   The flag resets to `False` when `_retinal_result_key = f"{label}_{confidence:.4f}"` detects a new scan, so auto-init fires normally for fresh uploads.

### Automated Navigation

After all models complete inference, I inject a zero-footprint JavaScript scroll via `st.components.v1.html(..., height=0)`:
```javascript
window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
```
This eliminates the need for the physician to scroll down to find results after clicking **Run Prediction** — a small but meaningful reduction in cognitive friction in a fast-paced clinical environment.

---

## 💡 Technical Philosophy

Through this project, I solidified my core philosophy as an AI Engineer: **In high-stakes environments, interpretability and reliability trump raw algorithmic complexity.** I don't just train models to hit a metric; I architect systems that solve real-world operational and human problems — using techniques like time-safe cumulative patient history, out-of-fold target encoding, Kansagara-grounded cost threshold optimization, and session-state-aware UI engineering — to ensure the output is as trustworthy and useful on day one in a hospital as it is on a benchmark leaderboard.
