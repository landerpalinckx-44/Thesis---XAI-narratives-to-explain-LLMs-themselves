# =============================================================================
# run_analysis.py
#
# STEP 1 of 3 — Run MExGen analysis and save results to disk.
#
# Usage:
#   python run_analysis.py                        # runs all instances
#   python run_analysis.py --instance 107         # runs a single instance by index
#   python run_analysis.py --force                # rerun even if results already exist
#
# Output:
#   mexgen_results/mexgen_results_instance_<index>.pkl per instance
# =============================================================================

import argparse
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from dotenv import load_dotenv

from icx360.algorithms.mexgen import CLIME
from icx360.utils.coloring_utils import color_units

import config
from models import OpenRouterModel
from prompts import build_gc_prompt

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run MExGen analysis for German Credit")
parser.add_argument("--instance", type=int,  default=None,  help="Single instance index to run (e.g. 107). If omitted, runs all instances.")
parser.add_argument("--force",    action="store_true",      help="Rerun even if results already exist")
args = parser.parse_args()

load_dotenv()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df_final_readable = pd.read_csv("selected_instances_readable.csv", index_col=0)

if "class" in df_final_readable.columns:
    df_final_readable = df_final_readable.drop(columns=["class"])

# Determine which instances to run
indices = [args.instance] if args.instance is not None else list(df_final_readable.index)

# ---------------------------------------------------------------------------
# Skip already-completed instances unless --force
# ---------------------------------------------------------------------------
os.makedirs(config.RESULTS_DIR, exist_ok=True)

if not args.force:
    indices = [
        idx for idx in indices
        if not os.path.exists(
            os.path.join(config.RESULTS_DIR, f"mexgen_results_instance_{idx}.pkl")
        )
    ]
    if not indices:
        print("All instances already analyzed. Nothing to do.")
        raise SystemExit(0)
    print(f"Skipping already completed instances. Remaining: {indices}")

# ---------------------------------------------------------------------------
# Setup MExGen explainer (instantiated once, reused across instances)
# ---------------------------------------------------------------------------
wrapped_model = OpenRouterModel(
    model_name=config.MEXGEN_MODEL,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=config.TEMPERATURE,
)

explainer = CLIME(
    wrapped_model,
    scalarizer="text",
    segmenter=config.SEGMENTER,
    sim_scores=[config.SCALARIZER],
)

# ---------------------------------------------------------------------------
# Loop over instances
# ---------------------------------------------------------------------------
for i, idx in enumerate(indices, start=1):
    print(f"\n{'='*60}")
    print(f"[{i}/{len(indices)}] Running instance index={idx}")
    print(f"{'='*60}")

    row = df_final_readable.loc[idx]
    applicant_profile = row.to_dict()

    system_prompt, prompt, unit_types, ind_segment = build_gc_prompt(applicant_profile)

    print("System prompt:\n", system_prompt)
    print("\nUser prompt:\n", "".join(prompt))

    # --- Sanity check: get original prediction ---
    output_orig = wrapped_model.generate(
        ["".join(prompt)],
        text_only=False,
        system_prompt=system_prompt,
    )
    print(f"\nOriginal prediction: {output_orig.output_text}")

    # Guard against None output before running MExGen
    if not output_orig.output_text or output_orig.output_text[0] is None:
        print(f"  Skipping instance {idx} — model returned None. Rerun with --instance {idx} --force")
        continue

    # --- Run MExGen (word-level, no sentence step for tabular features) ---
    print("\nRunning MExGen word-level explanation...")
    output_dict_words = explainer.explain_instance(
        prompt,
        unit_types,
        output_orig=output_orig,
        ind_segment=ind_segment,
        segment_type=config.SEGMENT_TYPE,
        model_params={"system_prompt": system_prompt, "max_tokens": 5},
    )

    # Normalize attribution scores to [-1, 1] (MaxAbs, preserves sign and zero)
    scores = output_dict_words["attributions"][config.SCALARIZER]
    max_abs = max(abs(v) for v in scores)
    if max_abs != 0:
        output_dict_words["attributions"][config.SCALARIZER] = [v / max_abs for v in scores]
        
    # --- Display results ---
    print("\n=== Word-level highlighted attributions ===")
    color_units(
        output_dict_words["attributions"]["units"],
        output_dict_words["attributions"][config.SCALARIZER],
    )

    print("\n=== Word-level attribution scores (sorted) ===")
    df_words = (
        pd.DataFrame(output_dict_words["attributions"])[
            ["units", config.SCALARIZER, "unit_types"]
        ]
        .sort_values(by=config.SCALARIZER, ascending=False)
    )
    print(df_words.to_string(index=False))

    # --- Save results ---
    filename = os.path.join(config.RESULTS_DIR, f"mexgen_results_instance_{idx}.pkl")
    mexgen_results = {
        "output_dict_words": output_dict_words,
        "system_prompt":     system_prompt,
        "prompt":            prompt,
        "unit_types":        unit_types,
        "ind_segment":       ind_segment,
        "instance_index":    idx,
        "applicant_profile": applicant_profile,
        "scalarizer":        config.SCALARIZER,
    }
    with open(filename, "wb") as f:
        pickle.dump(mexgen_results, f)

    print(f"\nSaved to '{filename}'")

print(f"\nDone. {len(indices)} instance(s) analyzed.")
