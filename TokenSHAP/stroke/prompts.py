# =============================================================================
# prompts.py — Prompt templates for the Stroke experiment
# =============================================================================

def build_stroke_prompt(patient_profile: str) -> tuple[str, str]:
    """
    Builds the stroke prediction prompt for a single patient instance.

    Returns a (system_prompt, user_prompt) tuple.
    - system_prompt: frozen instructions, feature reference, and formatting constraint. TokenSHAP never perturbs this.
    - user_prompt:   only the patient profile. TokenSHAP splits and masks tokens exclusively from this string.

    Parameters
    ----------
    patient_profile : str
        A string representation of a single row from df_final_readable.
    """

    system_prompt = """You are a stroke prediction model. Your task is to predict whether a patient is at high or low risk for a stroke based on their profile.
    Respond with only 'High' or 'Low'.

    Feature reference:
    - gender: 'Male', 'Female' or 'Other'
    - age: age of the patient
    - hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
    - heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
    - ever_married: 'No' or 'Yes'
    - work_type: 'children', 'Govt_job', 'Never_worked', 'Private' or 'Self-employed'
    - Residence_type: 'Rural' or 'Urban'
    - avg_glucose_level: average glucose level in blood
    - bmi: body mass index
    - smoking_status: 'formerly smoked', 'never smoked', 'smokes' or 'Unknown' """

    user_prompt = f"""Patient profile:
    {patient_profile}"""

    return system_prompt, user_prompt