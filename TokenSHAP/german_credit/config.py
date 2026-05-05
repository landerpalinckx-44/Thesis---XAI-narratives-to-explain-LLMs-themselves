# =============================================================================
# config.py — Central configuration for the German credit experiment
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
    "expert_judge": "You are a rigorous Expert Judge specializing in Explainable AI.",
    "bank_client":  "You are a client of a bank evaluating an explanation provided to you concerning your credit risk.",
    "bank_officer": "You are a bank officer evaluating the quality of an explanation provided to a client concerning their credit risk.",
}

# --- File Paths ---
RESULTS_DIR = "tokenshap_results"
LOG_JSON    = "german_credit_log.json"
