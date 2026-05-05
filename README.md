# Thesis: XAI narratives to explain LLMs themselves

A study of XAI narratives based on two explainability methods — **TokenSHAP** and **MExGen** — for generating and evaluating natural language explanations of LLMs decision-making.

## Overview

This repository implements an end-to-end pipeline for generating, evaluating, and statistically analysing natural language explanations produced by two distinct XAI attribution methods across two decision use cases:

| Use case | Task |
|---|---|
| German Credit | Loan application risk prediction |
| Stroke | Medical stroke risk prediction |

Explanations (narratives) are generated from feature attributions, scored by a panel of four LLM judges across three quality aspects, and analysed with non-parametric statistical tests.

## Repository Structure

```
Thesis/
├── TokenSHAP/          # TokenSHAP attribution method + experiments
├── MExGen/             # MExGen attribution method + experiments
└── Analysis/           # Cross-method statistical analysis
```

---

## TokenSHAP

Implements **TokenSHAP**: a Monte Carlo Shapley value estimator for token-level interpretability of language models. Importance scores are computed by measuring how each token contributes to the model output across random token subsets.

```
TokenSHAP/
├── token_shap/                         # Core library (TokenSHAP)
├── german_credit/                      # German credit experiment
│   ├── commands.md                     # Commands for running all 3 steps in the pipeline
│   ├── config.py                       # Model/judge/persona configuration
│   ├── models.py                       # Custom model wrapper for use with OpenRouter
│   ├── prompts.py                      # Domain-specific prompt builders
│   ├── run_analysis.py                 # Run the TokenSHAP explainability method
│   ├── token_shap_story.py             # Contains the prompts for generating the narrative
│   ├── run_narrative.py                # Generate narratives from attributions
│   ├── evaluation_prompts.py           # Contains the prompts for LLM-as-a-Judge
│   ├── run_evaluation.py               # Score narratives with LLM judges
│   ├── logger.py                       # Generates log of results from the experiments
│   ├── data_visualisation.ipynb        # Notebook containing code to generate various visualisations
│   ├── selected_instances_readable.csv # Contains the data of the selected instances
│   ├── tokenshap_results/              # Pickled attribution results (20 instances)
│   └── visualisations/                 # Shapley value plots + evaluation heatmaps
└── stroke/                             # Stroke experiment (identical structure)
```

---

## MExGen

Implements **MExGen**: a perturbation-based explainability method that masks input features and measures semantic similarity of outputs (via BERT embeddings) to estimate feature importance.

```
MExGen/
├── german_credit/
│   ├── commands.md                     # Commands for running all 3 steps in the pipeline
│   ├── config.py                       # Model/judge/persona configuration
│   ├── models.py                       # Custom model wrapper for use with OpenRouter
│   ├── prompts.py                      # Domain-specific prompt builders
│   ├── run_analysis.py                 # Run the MExGen explainability method
│   ├── mexgen_story.py                 # Contains the prompts for generating the narrative
│   ├── run_narrative.py                # Generate narratives from attributions
│   ├── evaluation_prompts.py           # Contains the prompts for LLM-as-a-Judge
│   ├── run_evaluation.py               # Score narratives with LLM judges
│   ├── logger.py                       # Generates log of results from the experiments
│   ├── data_visualisation.ipynb        # Notebook containing code to generate various visualisations
│   ├── selected_instances_readable.csv # Contains the data of the selected instances
│   ├── mexgen_results/                 # Pickled attribution results (10 instances)
│   └── visualisations/                 # Attribution plots + evaluation heatmaps
└── stroke/                             # Stroke experiment (identical structure)
```

---

## Analysis

Cross-method statistical analysis comparing TokenSHAP and MExGen explanations.

```
Analysis/
├── statistical_analysis.ipynb   # Main analysis notebook (8 sections)
├── BERTScore_test.ipynb         # BERTScore semantic similarity tests
├── analysis_outputs/
│   ├── summary.md               # Narrative summary of all findings
│   └── csv/                     # 30+ CSV files with test results
└── 33_judge_calibration_chart_stacked.svg
```

**Notebook sections:**

| Section | Content |
|---|---|
| 1 | Data overview — experiment shapes, score distributions |
| 2 | Human validation of LLM-as-Judge scores (Spearman correlation) |
| 3 | Judge agreement and calibration |
| 4 | Persona effects (Friedman + Wilcoxon post-hoc tests) |
| 5 | Evaluation metrics (aspect correlations, attribution concentration) |
| 6 | XAI method comparison — TokenSHAP vs MExGen (paired Wilcoxon) |
| 7 | Outlier analysis (Tukey + z-score) |
| 8 | TokenSHAP sampling cap analysis |

---

## Evaluation Framework

Both methods share the same evaluation pipeline:

- **Judges:** 4 LLMs (GPT, Gemini, Claude, Grok) via OpenRouter
- **Aspects:** Faithfulness, Helpfulness, Plausibility
- **Personas:** Role-based perspectives (expert judge, bank client, bank officer, doctor, patient, elaborate doctor, elaborate patient)
- **Aggregation:** Majority voting across judges

---

## Installation

**MExGen:** 
```bash
uv pip install icx360
```
**TokenSHAP:**

see `TokenSHAP/requirements.txt`
