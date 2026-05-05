# =============================================================================
# logger.py — Experiment logging for MExGen Stroke runs
# =============================================================================

import json
import os
import pandas as pd
from datetime import datetime


class NarrativeLogger:
    """
    Saves MExGen narratives and all relevant metadata to a JSON log file.
    Appends one entry per story — useful for comparing models, scalarizers,
    and temperature settings across runs.
    """

    def __init__(self, save_path: str = "stroke_log.json"):
        self.save_path = save_path
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                self.log = json.load(f)
        else:
            self.log = []

    def save(
        self,
        instance_index:    int,
        patient_profile:   dict,
        system_prompt:     str,
        prompt:            list[str],
        narrative_prompt:  str,
        mexgen_model:      str,
        scalarizer:        str,
        output_dict_words: dict,
        narrative_model:   str,
        temperature:       float,
        story:             str,
    ):
        # Extract model prediction
        answer = output_dict_words["output_orig"].output_text[0]

        # Serialize word-level attributions (exclude "n" units)
        words_df = pd.DataFrame(output_dict_words["attributions"])
        word_attributions = (
            words_df[words_df["unit_types"] != "n"][["units", scalarizer, "unit_types"]]
            .sort_values(by=scalarizer, ascending=False)
            .rename(columns={
                "units":      "unit",
                scalarizer:   "score",
                "unit_types": "granularity",
            })
            .to_dict(orient="records")
        )

        entry = {
            # --- Bookkeeping ---
            "timestamp":         datetime.now().isoformat(),
            "instance_index":    instance_index,
            "patient_profile":   patient_profile,

            # --- Prompt ---
            "system_prompt":     system_prompt,
            "prompt":            "".join(prompt),
            "narrative_prompt":  narrative_prompt,

            # --- MExGen config ---
            "mexgen_model":      mexgen_model,
            "scalarizer":        scalarizer,

            # --- MExGen results ---
            "baseline_response": answer,
            "word_attributions": word_attributions,

            # --- Narrative config ---
            "narrative_model":   narrative_model,
            "temperature":       temperature,

            # --- Narrative result ---
            "story":             story,
        }

        self.log.append(entry)

        with open(self.save_path, "w") as f:
            json.dump(self.log, f, indent=2)

        print(f"Saved entry #{len(self.log)} to '{self.save_path}'")
