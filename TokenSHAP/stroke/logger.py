# =============================================================================
# logger.py — Experiment logging for Stroke runs
# =============================================================================

import json
import os
from datetime import datetime


class NarrativeLogger:
    """
    Saves TokenSHAP narratives and all relevant metadata to a JSON log file.
    Appends one entry per story — useful for comparing models, prompts, and settings.
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
        prompt:           str,
        system_prompt:    str,
        narrative_prompt: str,
        instance_index:   int,
        patient_profile:  str,
        tokenshap_model:  str,
        sampling_ratio:   float,
        token_shap_instance,
        narrative_model:  str,
        temperature:      float,
        story:            str,
    ):
        entry = {
            "timestamp":         datetime.now().isoformat(),
            "instance_index":    instance_index,
            "patient_profile":   patient_profile,
            "system_prompt":     system_prompt,
            "prompt":            prompt,
            "narrative_prompt":  narrative_prompt,
            "tokenshap_model":   tokenshap_model,
            "sampling_ratio":    sampling_ratio,
            "baseline_response": token_shap_instance.baseline_text,
            "shapley_values":    {
                k: float(v) for k, v in token_shap_instance.shapley_values.items()
            },
            "narrative_model":   narrative_model,
            "temperature":       temperature,
            "story":             story,
        }

        self.log.append(entry)

        with open(self.save_path, "w") as f:
            json.dump(self.log, f, indent=2)

        print(f"Saved entry #{len(self.log)} to '{self.save_path}'")