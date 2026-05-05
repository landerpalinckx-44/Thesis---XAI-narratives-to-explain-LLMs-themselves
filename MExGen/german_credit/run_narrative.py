# =============================================================================
# run_narrative.py
#
# STEP 2 of 3 — Generate narratives from saved MExGen results.
#
# Usage:
#   python run_narrative.py                       # loops over all .pkl files in mexgen_results/
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

from dotenv import load_dotenv

import config
from logger import NarrativeLogger
from mexgen_story import MExGenStory

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Generate narratives from saved MExGen results")
parser.add_argument("--input",       type=str,   default=None,                       help="Single .pkl file. Omit to process all mexgen_results_*.pkl in mexgen_results/.")
parser.add_argument("--temperature", type=float, default=config.NARRATIVE_TEMPERATURE, help="Narrative generation temperature")
parser.add_argument("--log",         type=str,   default=config.LOG_JSON,            help="Path to JSON log file")
args = parser.parse_args()

load_dotenv()

# ---------------------------------------------------------------------------
# Collect files to process
# ---------------------------------------------------------------------------
if args.input is not None:
    pkl_files = [args.input]
else:
    pkl_files = sorted(
        glob.glob(os.path.join(config.RESULTS_DIR, "mexgen_results_instance_*.pkl"))
    )
    if not pkl_files:
        print(f"No mexgen_results_instance_*.pkl files found in '{config.RESULTS_DIR}/'. Run run_analysis.py first.")
        raise SystemExit(1)

print(f"Found {len(pkl_files)} file(s) to process: {pkl_files}")

# ---------------------------------------------------------------------------
# Setup — instantiated once, reused across all files
# ---------------------------------------------------------------------------
storyteller = MExGenStory(
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
        mexgen_results = pickle.load(f)

    output_dict_words = mexgen_results["output_dict_words"]
    system_prompt     = mexgen_results["system_prompt"]
    prompt            = mexgen_results["prompt"]
    scalarizer        = mexgen_results.get("scalarizer", config.SCALARIZER)

    print(f"Generating narrative (model={config.NARRATIVE_MODEL}, temp={args.temperature})...")
    story, narrative_prompt = storyteller.generate_story(
        output_dict_words=output_dict_words,
        scalarizer=scalarizer,
        system_prompt=system_prompt,
        original_prompt="".join(prompt),
        task_description="predict whether a loan applicant is a good or bad credit risk",
        temperature=args.temperature,
    )

    print("\n--- NARRATIVE ---\n")
    for sentence in story.split(". "):
        print(sentence.strip() + ".")
    print()

    logger.save(
        instance_index=mexgen_results.get("instance_index"),
        applicant_profile=mexgen_results.get("applicant_profile"),
        system_prompt=system_prompt,
        prompt=prompt,
        narrative_prompt=narrative_prompt,
        mexgen_model=config.MEXGEN_MODEL,
        scalarizer=scalarizer,
        output_dict_words=output_dict_words,
        narrative_model=config.NARRATIVE_MODEL,
        temperature=args.temperature,
        story=story,
    )

print(f"\nDone. {len(pkl_files)} narrative(s) logged to '{args.log}'.")
