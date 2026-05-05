# =============================================================================
# config.py — Central configuration for the Stroke experiment
# =============================================================================

# --- TokenSHAP Configuration ---
TOKENSHAP_MODEL = "meta-llama/llama-3.3-70b-instruct"
SAMPLING_RATIO  = 0.4   # ratio for monte carlo sampling in TokenSHAP (0.4 means 40% of all possible subsets are sampled)
                        # Starting from sampling ratio of 0.4, the increased number of combinations begins to reduce noise, 
                        # the larger the sampling ratio, the larger the accuracy compared to the actual shapley values.

# --- Narrative Configuration ---
NARRATIVE_MODEL       = "google/gemini-3-flash-preview"
NARRATIVE_TEMPERATURE = 1.0   # documentation says gemini 3 is optimized for temperature = 1

# --- Evaluation Configuration ---
JUDGE_MODELS = [
    "openai/gpt-5.4",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-sonnet-4.6",
    "x-ai/grok-4.20",
]
JUDGE_RUNS      = 5      # number of times each model runs each aspect (majority voting)
JUDGE_MAX_ATTEMPTS = 25                # maximum attempts per judge model to collect JUDGE_RUNS valid scores
JUDGE_ASPECTS   = ["faithfulness", "helpfulness", "plausibility"]

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
RESULTS_DIR = "tokenshap_results"
LOG_JSON    = "stroke_log.json"
