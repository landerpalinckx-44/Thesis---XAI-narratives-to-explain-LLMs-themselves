# =============================================================================
# run_analysis.py
#
# STEP 1 of 3 — Run TokenSHAP analysis and save results to disk.
#
# Usage:
#   python run_analysis.py                        # runs all instances
#   python run_analysis.py --instance 107         # runs a single instance by index
#   python run_analysis.py --sampling-ratio 0.5
#
# Output:
#   tokenshap_results_instance_<index>.pkl per instance
# =============================================================================

import argparse
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup — makes token_shap importable from the parent TokenSHAP folder
# ---------------------------------------------------------------------------
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from token_shap.base import TfidfTextVectorizer
from token_shap.token_shap import StringSplitter, TokenSHAP

import config
from models import OpenRouterModel
from prompts import build_gc_prompt

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run TokenSHAP analysis for German Credit")
parser.add_argument("--instance",       type=int,   default=None,                   help="Single instance index to run (e.g. 107). If omitted, runs all instances.")
parser.add_argument("--sampling-ratio", type=float, default=config.SAMPLING_RATIO,  help="TokenSHAP sampling ratio")
parser.add_argument("--force", action="store_true", help="Rerun even if results already exist")

args = parser.parse_args()

load_dotenv()

# ---------------------------------------------------------------------------
# Load the data — df_final_readable must be defined or imported
# ---------------------------------------------------------------------------
# If you have it saved as a CSV:
df_final_readable = pd.read_csv("selected_instances_readable.csv", index_col=0)

# Drop class column if it still exists (fail safe)
if 'class' in df_final_readable.columns:
    df_final_readable = df_final_readable.drop(columns=['class'])

# Determine which instances to run
if args.instance is not None:
    indices = [args.instance]
else:
    indices = list(df_final_readable.index)

# ---------------------------------------------------------------------------
# Skip already-completed instances unless --force
# ---------------------------------------------------------------------------
if not args.force:
    indices = [
        idx for idx in indices
        if not os.path.exists(os.path.join(config.RESULTS_DIR, f"tokenshap_results_instance_{idx}.pkl"))
    ]

    if not indices:
        print("All instances already analyzed. Nothing to do.")
        raise SystemExit(0)

    print(f"Skipping already completed instances. Remaining: {indices}")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
model = OpenRouterModel(
    model_name=config.TOKENSHAP_MODEL,
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
vectorizer = TfidfTextVectorizer()
splitter   = StringSplitter()
token_shap = TokenSHAP(model=model, splitter=splitter, vectorizer=vectorizer)

# ---------------------------------------------------------------------------
# Loop over instances
# ---------------------------------------------------------------------------
for i, idx in enumerate(indices, start=1):
    print(f"\n{'='*60}")
    print(f"[{i}/{len(indices)}] Running instance index={idx}")
    print(f"{'='*60}")

    # Format the row as a readable string
    row = df_final_readable.loc[idx]
    applicant_profile = "\n".join([f"  {col}: {val}" for col, val in row.items()])

    # Build the two-part prompt — system is frozen, user is perturbed
    system_prompt, user_prompt = build_gc_prompt(applicant_profile=applicant_profile)

    print("System prompt:\n", system_prompt)
    print("\nUser prompt:\n", user_prompt)

    token_shap.analyze(
        prompt=user_prompt,
        system_prompt=system_prompt,
        sampling_ratio=args.sampling_ratio,
    )

    # Display results
    token_shap.print_colored_text()
    print("\nShapley values:")
    print(token_shap.shapley_values)
    print("\nLLM Output:")
    print(token_shap.baseline_text)

    # Save results
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    filename = os.path.join(config.RESULTS_DIR, f"tokenshap_results_instance_{idx}.pkl")
    tokenshap_results = {
        "shapley_values":    token_shap.shapley_values,
        "baseline_text":     token_shap.baseline_text,
        "system_prompt":     system_prompt,
        "user_prompt":       user_prompt,
        "instance_index":    idx,
        "applicant_profile": applicant_profile,
        "sampling_ratio":    args.sampling_ratio,
    }
    with open(filename, "wb") as f:
        pickle.dump(tokenshap_results, f)

    print(f"Saved to '{filename}'")

print(f"\nDone. {len(indices)} instance(s) analyzed.")