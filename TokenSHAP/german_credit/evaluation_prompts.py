# =============================================================================
# evaluation_prompts.py — Judge prompts for narrative evaluation TokenSHAP
# =============================================================================

def build_faithfulness_prompt(
    persona_description: str,
    original_prompt:     str,
    model_choice:        str,
    shap_table:          str,
    narrative:           str,
) -> str:
    return f"""Role: {persona_description} Your task is to evaluate the Faithfulness of a human-readable narrative that explains an LLM's generated output.

    Inputs for Evaluation:
    - Original Prompt: {original_prompt}
    - Model Choice: {model_choice}
    - Shapley Value Table (sorted from most positively influential (top) to most negatively influential (bottom)):
    {shap_table}
    - Generated Narrative: {narrative}

    Evaluation Rubric: You must score the narrative from 1 (worst) to 5 (best) by following these evaluation steps:
    1. Read the narrative thoroughly.
    2. Take into account the faithfulness: Does the narrative accurately represent the model's actual decision-making process? It must align with the internal logic of the Shapley values provided in the table.
    Check: Are the tokens with the highest absolute Shapley values the ones highlighted in the story?
    3. Take into account the soundness: Is the information included in the narrative correct and non-misleading?
    Check: Does the story avoid misleading interpretations? For example, it should not refer to a low-impact word as "highly influential".
    4. Take into account factuality: Is the explanation strictly grounded in the provided attribution data?
    Check: Do not be fooled by "plausibility." Even if the story sounds believable or "natural," it is a failure if it is not grounded in the specific data points of the Shapley table.
    5. Provide a score.
    Only answer with a single score (between 1 and 5) without adding anything else."""


def build_helpfulness_prompt(
    persona_description: str, 
    original_prompt:     str, 
    model_choice:        str, 
    narrative:           str,
) -> str:
    return f"""Role: {persona_description} Your task is to evaluate the Helpfulness of a human-readable narrative that explains an LLM's generated output.

    Inputs for Evaluation:
    - Original Prompt: {original_prompt}
    - Model Choice: {model_choice}
    - Generated Narrative: {narrative}

    Evaluation Rubric: You must score the narrative from 1 (worst) to 5 (best) by following these evaluation steps:
    1. Read the narrative thoroughly.
    2. Take into account the Comprehensibility and Ease: Focus on the ease with which the narrative can be understood and the clarity with which it conveys the model's reasoning to humans.
    Check: Is the narrative easy for a non-technical person to interpret?
    3. Take into account the Usefulness: Does the explanation help the user better understand the LLM's specific decision?
    Check: Does the story effectively address the user's likely concern within the context of the original prompt?
    4. Take into account the Likeliness to Use/Value: Based on the experience of reading this narrative, is it a valuable addition to the LLM?
    Check: Is the narrative sufficiently informative and useful that a user would actually want to use it?
    5. Provide a score.
    Only answer with a single score (between 1 and 5) without adding anything else."""


def build_plausibility_prompt(
    persona_description: str,
    original_prompt:     str,
    model_choice:        str,
    narrative:           str,
) -> str:
    return f"""Role: {persona_description} Your task is to evaluate the Plausibility of a human-readable narrative that explains an LLM's generated output.

    Inputs for Evaluation:
    - Original Prompt: {original_prompt}
    - Model Choice: {model_choice}
    - Generated Narrative: {narrative}

    Evaluation Rubric: You must score the narrative from 1 (worst) to 5 (best) by following these evaluation steps:
    1. Read the narrative thoroughly.
    2. Take into account the Coherence and Fluency: Does the explanation fit naturally within the context and maintain grammatical consistency?
    Check: To what extent does the narrative sound "natural," like a human in conversation?
    3. Take into account Plausibility: Does the narrative appear reasonable, believable, and aligned with human expectations, independent of factual accuracy?
    Check: Does it align with domain knowledge and avoid internal contradictions or impossible claims?
    Check: Are any assumptions or general knowledge inclusions (outside the specific Shapley data) reasonable and logically consistent?
    Check: Does the narrative present internally coherent and correct reasoning?
    4. Take into account the Convincingness: Do you find this story provides a convincing interpretation of why the model made its prediction, independent of factual accuracy?
    5. Provide a score.
    Only answer with a single score (between 1 and 5) without adding anything else."""


ASPECT_PROMPT_BUILDERS = {
    "faithfulness":  build_faithfulness_prompt,
    "helpfulness":   build_helpfulness_prompt,
    "plausibility":  build_plausibility_prompt,
}