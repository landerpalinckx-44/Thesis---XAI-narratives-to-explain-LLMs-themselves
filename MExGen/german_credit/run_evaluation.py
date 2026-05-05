# =============================================================================
# run_evaluation.py
#
# STEP 3 of 3 — Evaluate narratives using LLM-as-a-Judge with majority voting.
#
# Usage:
#   python run_evaluation.py                      # evaluates all entries in german_credit_log.json
#   python run_evaluation.py --instance 107       # evaluates only entries for instance 107
#   python run_evaluation.py --temperature 0.0    # judge temperature (default 0.0 for consistency)
#
# Output:
#   Appends evaluation scores to each entry in german_credit_log.json
# =============================================================================

import argparse
import json
import os
from collections import Counter

from dotenv import load_dotenv
from openai import OpenAI

import config
from evaluation_prompts import ASPECT_PROMPT_BUILDERS

import pickle
from mexgen_story import MExGenStory

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate MExGen narratives using LLM-as-a-Judge")
parser.add_argument("--instance",    type=int,   default=None, help="Only evaluate entries for this instance index")
parser.add_argument("--temperature", type=float, default=0.0,  help="Judge model temperature (default 0.0)")
args = parser.parse_args()

load_dotenv()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

def query_judge(model: str, prompt: str, temperature: float) -> int | None:
    """Query a judge model and return an integer score 1-5, or None if parsing fails."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict evaluator. You must respond with a single integer between 1 and 5. No explanation, no reasoning, no other text.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=100,
        )
        raw = response.choices[0].message.content
        if raw is None:
            print(f"    Warning: {model} returned None content")
            return None
        raw = raw.strip()
        for token in raw.split():
            token_clean = token.strip(".,;:")
            if token_clean.isdigit():
                score = int(token_clean)
                if 1 <= score <= 5:
                    return score
        print(f"    Warning: {model} returned no valid score in: {raw[:80]}")
    except Exception as e:
        print(f"    Warning: {model} failed — {e}")
    return None


def query_judge_with_retry(model: str, prompt: str, temperature: float, target: int, max_attempts: int = config.JUDGE_MAX_ATTEMPTS) -> list[int]:
    """Keep querying until target valid scores are collected or max_attempts is reached."""
    scores   = []
    attempts = 0

    while len(scores) < target and attempts < max_attempts:
        attempts += 1
        score = query_judge(model, prompt, temperature)
        if score is not None:
            scores.append(score)
            print(f"      Attempt {attempts}: {score} ({len(scores)}/{target} valid)")
        else:
            print(f"      Attempt {attempts}: failed, retrying...")

    if len(scores) < target:
        print(f"    Warning: only {len(scores)}/{target} valid scores after {max_attempts} attempts")

    return scores


def majority_vote(scores: list[int]) -> int | float:
    """Return the most common score. Ties broken by the mean of tied scores."""
    count = Counter(scores)
    max_freq = count.most_common(1)[0][1]
    tied = [score for score, freq in count.items() if freq == max_freq]
    
    if len(tied) == 1:
        return tied[0]
    return sum(tied) / len(tied)


def evaluate_entry(entry: dict, temperature: float, persona: str, persona_description: str, attribution_table: str) -> dict:
    system_prompt      = entry.get("system_prompt", "")
    original_prompt    = entry["prompt"]
    model_choice       = entry["baseline_response"]
    narrative          = entry["story"]

    full_prompt = (
        f"System: {system_prompt}\n\nUser: {original_prompt}"
        if system_prompt else original_prompt
    )

    evaluation = {}

    for aspect in config.JUDGE_ASPECTS:
        print(f"  Aspect: {aspect}")
        builder = ASPECT_PROMPT_BUILDERS[aspect]

        if aspect == "faithfulness":
            judge_prompt = builder(persona_description, full_prompt, model_choice, attribution_table, narrative)
        else:
            judge_prompt = builder(persona_description, full_prompt, model_choice, narrative)

        aspect_results = {"judge_prompt": judge_prompt}

        for model in config.JUDGE_MODELS:
            print(f"    Judge: {model}")

            scores = query_judge_with_retry(model, judge_prompt, temperature, target=config.JUDGE_RUNS)

            if scores:
                final_score = majority_vote(scores)
                aspect_results[model] = {"runs": scores, "final_score": final_score}
                print(f"    → Majority vote: {final_score}")
            else:
                aspect_results[model] = {"runs": [], "final_score": None}

        # Aggregate across all judge models via majority vote
        all_final_scores = [
            v["final_score"]
            for k, v in aspect_results.items()
            if k != "judge_prompt" and v["final_score"] is not None
        ]
        aspect_results["aggregate_score"] = majority_vote(all_final_scores) if all_final_scores else None

        evaluation[aspect] = aspect_results

    return evaluation

storyteller = MExGenStory(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model_name=config.NARRATIVE_MODEL,
    base_url="https://openrouter.ai/api/v1",
)

# ---------------------------------------------------------------------------
# Load log
# ---------------------------------------------------------------------------
with open(config.LOG_JSON, "r") as f:
    log = json.load(f)

indices_to_evaluate = [
    i for i, entry in enumerate(log)
    if args.instance is None or entry.get("instance_index") == args.instance
]

if not indices_to_evaluate:
    print(f"No entries found for instance {args.instance}.")
    raise SystemExit(1)

print(f"Evaluating {len(indices_to_evaluate)} log entry/entries...")

# ---------------------------------------------------------------------------
# Evaluate and update log
# ---------------------------------------------------------------------------
for i, log_idx in enumerate(indices_to_evaluate, start=1):
    entry = log[log_idx]
    print(f"\n{'='*60}")
    print(f"[{i}/{len(indices_to_evaluate)}] Instance {entry.get('instance_index')} — log entry #{log_idx}")
    print(f"{'='*60}")

    entry.setdefault("evaluations", {})

    pkl_path = os.path.join(config.RESULTS_DIR, f"mexgen_results_instance_{entry['instance_index']}.pkl")
    with open(pkl_path, "rb") as f:
        mexgen_results = pickle.load(f)

    attribution_df = storyteller.build_attribution_df(
        mexgen_results["output_dict_words"],
        mexgen_results.get("scalarizer", config.SCALARIZER),
    )
    attribution_table = attribution_df.to_string(index=False)

    for persona, persona_description in config.JUDGE_PERSONAS.items():
        print(f"\n  --- Persona: {persona} ---")
        entry["evaluations"][persona] = evaluate_entry(
            entry, args.temperature, persona, persona_description, attribution_table
        )

    # Save after each entry so progress isn't lost
    with open(config.LOG_JSON, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Saved evaluation to '{config.LOG_JSON}'")

print(f"\nDone. {len(indices_to_evaluate)} entry/entries evaluated.")
