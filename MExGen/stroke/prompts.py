# =============================================================================
# prompts.py — Prompt templates for the MExGen Stroke experiment
# =============================================================================


def build_stroke_prompt(patient_profile: dict) -> tuple[str, list[str], list[str], list[bool]]:
    """
    Builds the stroke prediction prompt in MExGen list format for a single patient.

    Returns a tuple of:
    - system_prompt : frozen instructions passed to the model as a system message.
                      MExGen never perturbs this.
    - prompt        : list of prompt elements. Each feature is its own element so
                      MExGen can attribute to it individually.
    - unit_types    : parallel list marking each element as "p" (to be attributed)
                      or "n" (not of interest / never perturbed).
    - ind_segment   : parallel bool list marking which elements to segment into
                      words. True for all feature lines, False for structural text.

    Parameters
    ----------
    patient_profile : dict
        A dict of {feature_name: value} for a single row from df_final_readable.
    """

    system_prompt = (
        "You are a stroke prediction model. Your task is to predict whether a patient "
        "is at high or low risk for a stroke based on their profile. "
        "Respond with only 'High' or 'Low'.\n\n"
        "Feature reference:\n"
        "- gender: 'Male', 'Female' or 'Other'\n"
        "- age: age of the patient\n"
        "- hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension\n"
        "- heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease\n"
        "- ever_married: 'No' or 'Yes'\n"
        "- work_type: 'children', 'Govt_job', 'Never_worked', 'Private' or 'Self-employed'\n"
        "- Residence_type: 'Rural' or 'Urban'\n"
        "- avg_glucose_level: average glucose level in blood\n"
        "- bmi: body mass index\n"
        "- smoking_status: 'formerly smoked', 'never smoked', 'smokes' or 'Unknown'"
    )

    # Build feature lines from the profile dict
    feature_lines = [f"{col}: {val}\n" for col, val in patient_profile.items()]

    # Structural elements (not attributed to)
    prefix   = "Patient profile:\n"

    prompt = [prefix] + feature_lines

    unit_types = (
        ["n"]                         # prefix
        + ["p"] * len(feature_lines)  # one "p" per feature
    )

    ind_segment = (
        [False]                       # prefix — don't segment
        + [True] * len(feature_lines) # segment each feature into words
    )

    return system_prompt, prompt, unit_types, ind_segment
