# =============================================================================
# config.py — Central configuration for the MExGen Stroke experiment
# =============================================================================

# --- MExGen Configuration ---
MEXGEN_MODEL   = "meta-llama/llama-3.3-70b-instruct"     # LLM used by MExGen for perturbation-based attribution
SCALARIZER     = "bert"                     # similarity metric used by MExGen ("bert" or "st")
SEGMENTER      = "en_core_web_sm"           # spaCy model used for word/sentence segmentation
SEGMENT_TYPE   = "w"                        # "w" = word-level (skip sentence step for tabular features)
NUM_TOP_UNITS  = 1                          # how many top units to refine into words (if doing two-stage)
TEMPERATURE    = 0.6                        # LLM temperature during MExGen perturbations (same as TokenSHAP for consistency)

# --- Narrative Configuration ---
NARRATIVE_MODEL       = "google/gemini-3-flash-preview"
NARRATIVE_TEMPERATURE = 1.0                 # documentation says gemini 3 is optimized for temperature = 1

# --- Evaluation Configuration ---
JUDGE_MODELS = [
    "openai/gpt-5.4",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-sonnet-4.6",
    "x-ai/grok-4.20",
]
JUDGE_RUNS    = 5                           # runs per judge model per aspect (majority voting)
JUDGE_MAX_ATTEMPTS = 25                # maximum attempts per judge model to collect JUDGE_RUNS valid scores
JUDGE_ASPECTS = ["faithfulness", "helpfulness", "plausibility"]

JUDGE_PERSONAS = {
    # simple personas
    "expert_judge": "You are a rigorous Expert Judge specializing in Explainable AI.",
    "patient":      "You are a patient who has inquired about your stroke risk and are evaluating the explanation provided to you.",
    "doctor":       "You are a doctor evaluating the quality of an explanation provided to a patient concerning their stroke risk.",
    # elaborate personas to counter persona sparsity issue:
    "patient_elaborate": """Given a {age}-year-old {gender} patient who {ever_married}, works in the 
    {work_type} sector and lives in a {Residence_type} area. This patient has been 
    told they are at {risk} risk for a stroke. Their relevant medical history includes: 
    {hypertension_str}, {heart_disease_str}, a BMI of {bmi}, an average glucose level 
    of {avg_glucose_level}, and they {smoking_str}. They have limited medical knowledge. 
    How would this patient rate the following explanation concerning their stroke risk?""",

    "doctor_elaborate": "Given a doctor with extensive medical knowledge of stroke risk factors — "
    "including the clinical significance of glucose levels, BMI, smoking history, "
    "and cardiovascular conditions — but limited technical knowledge of AI systems and explainability methods. "
    "This doctor regularly communicates risk assessments to patients with varying levels of health literacy. "
    "How would this doctor rate the following explanation provided to a patient concerning their stroke risk?"
}

# --- File Paths ---
RESULTS_DIR = "mexgen_results"
LOG_JSON    = "stroke_log.json"
