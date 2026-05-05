# =============================================================================
# config.py — Central configuration for the MExGen German credit experiment
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
    "expert_judge": "You are a rigorous Expert Judge specializing in Explainable AI.",
    "bank_client":  "You are a client of a bank evaluating an explanation provided to you concerning your credit risk.",
    "bank_officer": "You are a bank officer evaluating the quality of an explanation provided to a client concerning their credit risk.",
}

# --- File Paths ---
RESULTS_DIR = "mexgen_results"
LOG_JSON    = "german_credit_log.json"
