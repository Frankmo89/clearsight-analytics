"""
Shared Data Pipeline — ClearSight Analytics
=============================================
Data loading, cleaning, feature engineering, and splitting for Models 1 & 2.
Reused in both train.py and predict.py to ensure consistent preprocessing.

Usage:
    from pipelines.data_pipeline import load_and_clean, engineer_features, split_data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pipelines.preprocessing_hints import create_medication_categories, preprocess_review_text

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# =============================================================================
# STEP 1: Load and Clean
# =============================================================================

def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load patient encounters CSV and apply all cleaning steps.
    Works on both training data and unseen test data.
    """
    df = pd.read_csv(filepath)

    # 1. Replace '?' with NaN (hidden missing values)
    df.replace('?', np.nan, inplace=True)

    # 2. Exclude death/hospice patients (cannot be readmitted = data leakage)
    exclude_dispositions = [11, 13, 14, 19, 20, 21]
    df = df[~df['discharge_disposition_id'].isin(exclude_dispositions)]

    # 3. Create binary readmission target
    #    '<30' and '>30' -> 1 (readmitted), 'NO' -> 0 (not readmitted)
    if 'readmitted' in df.columns:
        df['readmission_binary'] = (df['readmitted'] != 'NO').astype(int)

    # 4. Convert age brackets to numeric midpoints
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25,
        '[30-40)': 35, '[40-50)': 45, '[50-60)': 55,
        '[60-70)': 65, '[70-80)': 75, '[80-90)': 85,
        '[90-100)': 95
    }
    df['age_numeric'] = df['age'].map(age_map)

    # 5. Patient history features (computed BEFORE dropping patient_nbr)
    # -------------------------------------------------------------------------
    # encounter_id is sequential in time per UCI dataset documentation, so we
    # sort by it to ensure each encounter only "sees" its actual past.
    # cumcount() gives 0 for the first encounter, 1 for the second, etc.
    # cumsum().shift(1) shifts so a row never includes itself.
    # -------------------------------------------------------------------------
    if 'patient_nbr' in df.columns and 'encounter_id' in df.columns:
        df = df.sort_values('encounter_id').reset_index(drop=True)

        # How many prior encounters does this patient have in the dataset?
        df['prior_encounters_count'] = df.groupby('patient_nbr').cumcount()

        # Binary flag: is this a recurrent patient?
        df['is_recurrent_patient'] = (df['prior_encounters_count'] > 0).astype(int)

        # Cumulative prior inpatient visits (strongest historical signal).
        # shift(1) within group ensures we exclude the current row.
        df['prior_inpatient_cumsum'] = (
            df.groupby('patient_nbr')['number_inpatient']
            .apply(lambda x: x.shift(1).cumsum())
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
    else:
        df['prior_encounters_count'] = 0
        df['is_recurrent_patient'] = 0
        df['prior_inpatient_cumsum'] = 0

    # 6. Drop columns we won't use
    drop_cols = [
        # encounter_id kept here intentionally — prepare_test_data() captures
        # real IDs from df before engineer_features() drops the column.
        'patient_nbr',    # patient ID, not a feature (history features already computed)
        'weight',         # 97% missing
        'payer_code',     # 40% missing, not clinically useful
        'readmitted',     # replaced by readmission_binary
        'age',            # replaced by age_numeric
        'examide',        # zero variance
        'citoglipton',    # zero variance
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    print(f"Loaded and cleaned: {len(df)} rows, {len(df.columns)} columns")
    return df


# =============================================================================
# STEP 2: Feature Engineering
# =============================================================================

def categorize_icd9(code) -> str:
    """Group ICD-9 diagnostic codes into clinical categories."""
    if pd.isna(code):
        return 'missing'
    code = str(code).strip()

    if code.startswith('250'):
        return 'diabetes'
    try:
        numeric = float(code[:3])
    except ValueError:
        if code.startswith('V') or code.startswith('E'):
            return 'external'
        return 'other'

    if 390 <= numeric <= 459:
        return 'circulatory'
    elif 460 <= numeric <= 519:
        return 'respiratory'
    elif 520 <= numeric <= 579:
        return 'digestive'
    elif 580 <= numeric <= 629:
        return 'genitourinary'
    elif 710 <= numeric <= 739:
        return 'musculoskeletal'
    elif 800 <= numeric <= 999:
        return 'injury'
    else:
        return 'other'

def bin_length_of_stay(days) -> int:
    """
    Bin time_in_hospital into clinical risk tiers.

    Clinical rationale: length-of-stay risk curve is non-linear.
    - 1-2 days: short stay, usually lower risk
    - 3-5 days: standard stay
    - 6+ days: extended stay, often signals complications or unstable patient
    XGBoost can split on raw values, but the DNN (with StandardScaler)
    benefits from explicit ordinal bins.
    """
    if pd.isna(days):
        return 1  # default to mid tier
    if days <= 2:
        return 0
    elif days <= 5:
        return 1
    else:
        return 2



def _target_encode_column(df, col, target_col='readmission_binary',
                          preprocessing_state=None, state_key=None,
                          n_splits=5, random_state=42):
    """
    Apply target encoding to a single column with out-of-fold CV to prevent leakage.

    Training mode (preprocessing_state is None):
        - Computes 5-fold out-of-fold means for training rows
        - Stores the full target mean map + global mean for later use
        - Returns modified df and updated state dict

    Prediction mode (preprocessing_state provided):
        - Applies the saved map, filling unseen categories with global mean

    Args:
        df: DataFrame with the column to encode
        col: name of the column to encode (will be replaced by {col}_te)
        target_col: target variable name
        preprocessing_state: dict with saved encoders/maps (None during training)
        state_key: key under which the map is saved in preprocessing_state
        n_splits: CV folds for out-of-fold encoding (default 5)

    Returns:
        df with {col}_te added and original {col} dropped
        encoding_info dict: {'map': {...}, 'global': float}
    """
    from sklearn.model_selection import KFold

    # Ensure string type for consistent grouping (handles mix of int/str)
    df[col] = df[col].fillna('Unknown').astype(str)

    if preprocessing_state is None:
        # TRAINING MODE: compute out-of-fold encoding
        global_mean = df[target_col].mean()

        # OOF encoding for training rows
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        oof_encoded = pd.Series(index=df.index, dtype=float)
        for train_idx, val_idx in kf.split(df):
            fold_map = df.iloc[train_idx].groupby(col)[target_col].mean()
            oof_encoded.iloc[val_idx] = (
                df.iloc[val_idx][col].map(fold_map).fillna(global_mean)
            )

        # Full map for prediction time
        full_map = df.groupby(col)[target_col].mean().to_dict()
        df[f'{col}_te'] = oof_encoded
        encoding_info = {'map': full_map, 'global': global_mean}
    else:
        # PREDICTION MODE: apply saved map
        saved = preprocessing_state.get(state_key, {'map': {}, 'global': 0.5})
        df[f'{col}_te'] = df[col].map(saved['map']).fillna(saved['global'])
        encoding_info = saved

    # Drop the raw column (replaced by _te version)
    df.drop(columns=[col], inplace=True)
    return df, encoding_info


def engineer_features(df: pd.DataFrame, preprocessing_state=None):
    """
    Create all engineered features for Models 1 & 2.
    Must work on both training data and unseen test data.

    Training mode (preprocessing_state=None): fits encoders, computes medians,
    builds target encoding map, and returns the state dict.

    Prediction mode (preprocessing_state provided): uses saved encoders,
    medians, and target encoding map for consistent transforms.
    """
    # Drop encounter_id before feature engineering so it never becomes a feature.
    # prepare_test_data() captures IDs from df *before* calling this function.
    if 'encounter_id' in df.columns:
        df = df.drop(columns=['encounter_id'])


    # --- Medication features ---
    med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                'miglitol', 'troglitazone', 'tolazamide', 'insulin',
                'glyburide-metformin', 'glipizide-metformin',
                'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                'metformin-pioglitazone']
    med_cols = [c for c in med_cols if c in df.columns]

    df['n_meds_changed'] = df[med_cols].apply(
        lambda row: sum(1 for v in row if v != 'No'), axis=1
    )
    df['any_med_changed'] = (df['n_meds_changed'] > 0).astype(int)
    df['n_meds_increased'] = df[med_cols].apply(
        lambda row: sum(1 for v in row if v == 'Up'), axis=1
    )
    df['n_meds_decreased'] = df[med_cols].apply(
        lambda row: sum(1 for v in row if v == 'Down'), axis=1
    )

    # NEW: insulin usage flag (captured before we drop the med columns)
    #      Insulin is one of the strongest readmission predictors in literature.
    df['on_insulin'] = (df['insulin'] != 'No').astype(int) if 'insulin' in df.columns else 0

    # Drop individual medication columns (replaced by aggregates)
    df.drop(columns=med_cols, inplace=True)

    # --- Clinical test missingness flags ---
    df['glucose_tested'] = (df['max_glu_serum'] != 'None').astype(int) if 'max_glu_serum' in df.columns else 0
    df['a1c_tested'] = (df['A1Cresult'] != 'None').astype(int) if 'A1Cresult' in df.columns else 0

    # Encode test results as ordinal
    glu_map = {'None': 0, 'Norm': 1, '>200': 2, '>300': 3}
    a1c_map = {'None': 0, 'Norm': 1, '>7': 2, '>8': 3}
    if 'max_glu_serum' in df.columns:
        df['max_glu_serum'] = df['max_glu_serum'].map(glu_map).fillna(0).astype(int)
    if 'A1Cresult' in df.columns:
        df['A1Cresult'] = df['A1Cresult'].map(a1c_map).fillna(0).astype(int)

    # --- ICD-9 diagnosis categories ---
    for diag_col in ['diag_1', 'diag_2', 'diag_3']:
        if diag_col in df.columns:
            df[f'{diag_col}_cat'] = df[diag_col].apply(categorize_icd9)
            df.drop(columns=[diag_col], inplace=True)

    # --- Clinical complexity score ---
    df['complexity_score'] = (
        df['num_lab_procedures'].fillna(0) / 50 +
        df['num_procedures'].fillna(0) / 6 +
        df['num_medications'].fillna(0) / 20 +
        df['number_diagnoses'].fillna(0) / 9 +
        df['number_emergency'].fillna(0) / 3 +
        df['number_inpatient'].fillna(0) / 5
    )

    # =========================================================================
    # NEW ENHANCED FEATURES (FASE 1)
    # =========================================================================

    # --- Feature 1: Total prior hospital utilization ---
    # Clinical rationale: captures overall healthcare system engagement.
    # A patient with many prior visits of any type is systemically sicker.
    df['total_prior_visits'] = (
        df['number_outpatient'].fillna(0)
        + df['number_emergency'].fillna(0)
        + df['number_inpatient'].fillna(0)
    )

    # --- Feature 2: Service utilization ratio ---
    # Clinical rationale: distinguishes patients who rely on ER/hospital
    # (high-risk pattern) vs those with regular outpatient care (low-risk pattern).
    # +1 in denominator avoids division by zero.
    df['service_utilization_ratio'] = (
        df['number_emergency'].fillna(0) + df['number_inpatient'].fillna(0)
    ) / (df['number_outpatient'].fillna(0) + 1)

    # --- Feature 3: Diagnosis density (diagnoses per day of stay) ---
    # Clinical rationale: many diagnoses in a short stay = clinically unstable
    # patient. +1 avoids division by zero for same-day discharges.
    df['diagnoses_per_day'] = df['number_diagnoses'].fillna(0) / (
        df['time_in_hospital'].fillna(1) + 1
    )

    # --- Feature 4: Length-of-stay tier (ordinal bin) ---
    # Non-linear risk: 1-2 days, 3-5 days, 6+ days.
    df['los_tier'] = df['time_in_hospital'].apply(bin_length_of_stay).astype(int)

    # --- Feature 5: Insulin x medication-change interaction ---
    # Clinical rationale: a patient on insulin whose regimen was changed during
    # this encounter is at elevated readmission risk. 
    
    df['on_insulin'] = pd.to_numeric(df['on_insulin'], errors='coerce').fillna(0)
    
    if df['change'].dtype == 'object':
        change_flag = (df['change'] == 'Ch').astype(int)
    else:
        change_flag = pd.to_numeric(df['change'], errors='coerce').fillna(0).astype(int)

    df['insulin_and_change'] = (df['on_insulin'] * change_flag).astype(int)

    # --- Feature 6: High-complexity flag (binary) ---
    # Complement to continuous complexity_score; gives tree models a clean split.
    # Threshold computed from training distribution and saved in preprocessing_state.
    if preprocessing_state is None:
        complexity_threshold = df['complexity_score'].quantile(0.75)
    else:
        complexity_threshold = preprocessing_state.get('complexity_threshold', 1.5)
    df['high_complexity_flag'] = (df['complexity_score'] >= complexity_threshold).astype(int)

    # --- Encode change and diabetesMed as binary ---
    if 'change' in df.columns and df['change'].dtype == 'object':
        df['change'] = (df['change'] == 'Ch').astype(int)
    if 'diabetesMed' in df.columns:
        df['diabetesMed'] = (df['diabetesMed'] == 'Yes').astype(int)

    # --- Target encoding for high-signal categorical columns ---
    # LabelEncoder assigns arbitrary ordinal codes (noise). Target encoding
    # replaces each category with its out-of-fold mean readmission rate.
    # SHAP analysis showed these 7 categoricals carry meaningful signal:
    # medical_specialty (top 5), admission_source_id, diag_1_cat,
    # discharge_disposition_id, admission_type_id, diag_2_cat, diag_3_cat.
    #
    # Each uses 5-fold CV to prevent leakage (a row never sees its own target).
    target_encoded_cols = [
        'medical_specialty',
        'admission_source_id',
        'discharge_disposition_id',
        'admission_type_id',
        'diag_1_cat',
        'diag_2_cat',
        'diag_3_cat',
    ]

    # Store target encoding info per column in the state dict
    if preprocessing_state is None:
        te_info = {}
        if 'readmission_binary' in df.columns:
            for col in target_encoded_cols:
                if col in df.columns:
                    df, info = _target_encode_column(
                        df, col,
                        target_col='readmission_binary',
                        preprocessing_state=None,
                        state_key=f'te_{col}',
                    )
                    te_info[f'te_{col}'] = info
        else:
            # Edge case: training without target — should not happen in normal flow
            for col in target_encoded_cols:
                if col in df.columns:
                    df[f'{col}_te'] = 0.5
                    df.drop(columns=[col], inplace=True)
    else:
        for col in target_encoded_cols:
            if col in df.columns:
                df, _ = _target_encode_column(
                    df, col,
                    target_col='readmission_binary',
                    preprocessing_state=preprocessing_state,
                    state_key=f'te_{col}',
                )

    # --- Encode remaining categorical columns with LabelEncoder ---
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ['readmitted']]

    if preprocessing_state is None:
        # TRAINING MODE: fit new encoders and save medians
        encoders = {}
        for col in cat_cols:
            df[col] = df[col].fillna('Unknown')
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        # Save medians from training data for consistent imputation
        medians = {}
        for col in df.select_dtypes(include='number').columns:
            if df[col].isna().any():
                medians[col] = df[col].median()
                df[col] = df[col].fillna(medians[col])

        preprocessing_state = {
            'encoders': encoders,
            'medians': medians,
            'complexity_threshold': complexity_threshold,
        }
        # Add all target encoding maps we computed above
        preprocessing_state.update(te_info)
    else:
        # PREDICTION MODE: use saved encoders and medians
        encoders = preprocessing_state['encoders']
        medians = preprocessing_state['medians']

        for col in cat_cols:
            df[col] = df[col].fillna('Unknown')
            if col in encoders:
                le = encoders[col]
                df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            else:
                df[col] = 0

        # Fill NaN using training medians (not test medians)
        for col, median_val in medians.items():
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(median_val)

    # Fill any column still missing (edge case) with 0
    df.fillna(0, inplace=True)

    print(f"Feature engineering complete: {len(df.columns)} columns")
    return df, preprocessing_state


# =============================================================================
# STEP 3: Split Data
# =============================================================================

def split_data(df: pd.DataFrame, target_col: str = 'readmission_binary',
               test_size: float = 0.2, random_state: int = 42):
    """
    Split into X_train, X_val, y_train, y_val with stratification.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train: {len(X_train)} rows | Val: {len(X_val)} rows")
    print(f"Train target distribution:\n{y_train.value_counts(normalize=True).round(3)}")
    return X_train, X_val, y_train, y_val


# =============================================================================
# STEP 4: Full Pipeline (convenience function)
# =============================================================================

def prepare_training_data(filepath: str):
    """
    Run the full pipeline: load → clean → engineer → split.
    Used by train.py for both Model 1 and Model 2.

    Returns: X_train, X_val, y_train, y_val, preprocessing_state
    """
    df = load_and_clean(filepath)
    df, preprocessing_state = engineer_features(df)
    X_train, X_val, y_train, y_val = split_data(df)
    return X_train, X_val, y_train, y_val, preprocessing_state


def prepare_test_data(filepath: str, preprocessing_state: dict):
    """
    Run the pipeline WITHOUT splitting (for predict.py on unseen data).
    Uses preprocessing_state from training for consistent encoding and imputation.

    Returns:
        X: feature matrix ready for prediction
        ids: encounter IDs aligned with X rows (for output CSV)
    """
    df = load_and_clean(filepath)

    # Capture encounter IDs AFTER load_and_clean (which sorts by encounter_id
    # and filters death/hospice rows). This ensures IDs are aligned with
    # the feature rows that engineer_features will produce.
    if 'encounter_id' in df.columns:
        ids = df['encounter_id'].values.copy()
    else:
        ids = np.arange(len(df))

    df, _ = engineer_features(df, preprocessing_state=preprocessing_state)

    if 'readmission_binary' in df.columns:
        df.drop(columns=['readmission_binary'], inplace=True)

    print(f"Test data prepared: {len(df)} rows, {len(df.columns)} columns")
    return df, ids


# =============================================================================
# NLP PIPELINE — Model 4 (Wes)
# Handles the patient medication review text dataset.
# =============================================================================

def load_raw_data(filename: str) -> pd.DataFrame:
    """Load a raw CSV file from data/raw/.

    Args:
        filename: Name of the CSV file (e.g., "patient_medication_feedback.csv")

    Returns:
        pandas DataFrame
    """
    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Make sure you've downloaded the data to data/raw/"
        )
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filename}")
    return df


def clean_nlp_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the patient medication feedback dataset.

    - Drops rows missing the review text or target label
    - Removes rows where the review text is empty/whitespace
    - Resets the index

    Returns:
        Cleaned DataFrame
    """
    target_col = "effectiveness_3class"
    text_col = "benefitsReview"

    df = df.dropna(subset=[text_col, target_col])
    df = df[df[text_col].str.strip().str.len() > 0]
    df = df.reset_index(drop=True)

    print(f"After cleaning: {len(df)} rows")
    print(df[target_col].value_counts())
    return df


def engineer_nlp_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features useful for analysis alongside the NLP model.

    - review_word_count: number of words in the review
    - review_char_count: number of characters in the review
    - review_text_clean: LSTM-ready text (lowercase, letters only, no punctuation/numbers)
                         benefitsReview is kept raw for BioBERT

    Returns:
        DataFrame with added feature columns
    """
    df["review_word_count"] = df["benefitsReview"].str.split().str.len()
    df["review_char_count"] = df["benefitsReview"].str.len()
    df["review_text_clean"] = df["benefitsReview"].apply(preprocess_review_text)
    return df


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """Save processed data to data/processed/.

    Args:
        df: Processed DataFrame
        filename: Output filename (e.g., "nlp_processed.csv")
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")


def load_processed_data(filename: str) -> pd.DataFrame:
    """Load previously processed data from data/processed/.

    Args:
        filename: Name of the processed CSV file

    Returns:
        pandas DataFrame
    """
    filepath = PROCESSED_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data not found: {filepath}\n"
            f"Run the data pipeline first to generate processed data."
        )
    return pd.read_csv(filepath)
