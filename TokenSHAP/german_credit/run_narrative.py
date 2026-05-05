# =============================================================================
# run_narrative.py
#
# STEP 2 of 3 — Generate narratives from saved TokenSHAP results.
#
# Usage:
#   python run_narrative.py                       # loops over all .pkl files in current dir
#   python run_narrative.py --input my_run.pkl    # single file
#   python run_narrative.py --temperature 0.8
#
# Output:
#   Prints each story to console and appends all entries to german_credit_log.json
# =============================================================================

import argparse
import glob
import os
import pickle
from types import SimpleNamespace

from dotenv import load_dotenv

import config
from logger import NarrativeLogger
from token_shap_story import TokenSHAPstory

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Generate narratives from saved TokenSHAP results")
parser.add_argument("--input",       type=str,   default=None,               help="Single .pkl file. Omit to process all tokenshap_results_*.pkl in current dir.")
parser.add_argument("--temperature", type=float, default=config.NARRATIVE_TEMPERATURE, help="Narrative generation temperature")
parser.add_argument("--log",         type=str,   default=config.LOG_JSON,    help="Path to JSON log file")
args = parser.parse_args()

load_dotenv()

# Collect files to process
if args.input is not None:
    pkl_files = [args.input]
else:
    pkl_files = sorted(glob.glob(os.path.join(config.RESULTS_DIR, "tokenshap_results_instance_*.pkl")))
    if not pkl_files:
        print("No tokenshap_results_*.pkl files found. Run run_analysis.py first.")
        raise SystemExit(1)

print(f"Found {len(pkl_files)} file(s) to process: {pkl_files}")

# ---------------------------------------------------------------------------
# Setup — instantiated once, reused across all files
# ---------------------------------------------------------------------------
storyteller = TokenSHAPstory(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model_name=config.NARRATIVE_MODEL,
    base_url="https://openrouter.ai/api/v1",
)
logger = NarrativeLogger(save_path=args.log)

# ---------------------------------------------------------------------------
# Loop over .pkl files
# ---------------------------------------------------------------------------
for i, pkl_path in enumerate(pkl_files, start=1):
    print(f"\n{'='*60}")
    print(f"[{i}/{len(pkl_files)}] {pkl_path}")
    print(f"{'='*60}")

    with open(pkl_path, "rb") as f:
        tokenshap_results = pickle.load(f)

    # SimpleNamespace lets us access dict keys as attributes (.shapley_values, .baseline_text)
    # so TokenSHAPstory.generate_story() works without modification
    token_shap = SimpleNamespace(**tokenshap_results)
    prompt = tokenshap_results["user_prompt"]

    print(f"Generating narrative (model={config.NARRATIVE_MODEL}, temp={args.temperature})...")
    story, narrative_prompt = storyteller.generate_story(
        token_shap_instance=token_shap,
        original_prompt=prompt,
        system_prompt=tokenshap_results.get("system_prompt", ""),
        task_description="predict whether a loan applicant is a good or bad credit risk",
        temperature=args.temperature,
    )

    print("\n--- NARRATIVE ---\n")
    for sentence in story.split(". "):
        print(sentence.strip() + ".")
    print()

    logger.save(
        prompt=prompt,
        system_prompt=tokenshap_results.get("system_prompt"),
        narrative_prompt=narrative_prompt,
        instance_index=tokenshap_results.get("instance_index"),       
        applicant_profile=tokenshap_results.get("applicant_profile"),  
        tokenshap_model=config.TOKENSHAP_MODEL,
        sampling_ratio=tokenshap_results.get("sampling_ratio", config.SAMPLING_RATIO),
        token_shap_instance=token_shap,
        narrative_model=config.NARRATIVE_MODEL,
        temperature=args.temperature,
        story=story,
    )

print(f"\nDone. {len(pkl_files)} narrative(s) logged to '{args.log}'.")