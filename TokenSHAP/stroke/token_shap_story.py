# =============================================================================
# token_shap_story.py — TokenSHAPstory class for stroke narrative generation
# =============================================================================

import pandas as pd
from openai import OpenAI


class TokenSHAPstory:
    """
    Generates narratives explaining TokenSHAP results using an LLM.
    Analogous to SHAPstory but for prompt-level token attribution.
    """

    def __init__(self, api_key: str, model_name: str, base_url: str = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def build_shap_df(self, token_shap_instance) -> pd.DataFrame:
        """
        Converts TokenSHAP results into a DataFrame mirroring the tabular SHAP format.
        """
        shap_vals = token_shap_instance.shapley_values  # dict: {"token_index": value}
        rows = []
        for key, shap_value in shap_vals.items():
            parts = key.rsplit("_", 1)
            token_text = parts[0]
            rows.append({
                "token":       key,
                "token": token_text,
                "shap_value":  float(shap_value),
            })
        return pd.DataFrame(rows).sort_values("shap_value", ascending=False)[["token", "shap_value"]]

    def generate_narrative_prompt(
        self,
        original_prompt:  str,
        system_prompt:    str,
        model_response:   str,
        shap_df:          pd.DataFrame,
        task_description: str,
    ) -> str:
        shap_table = shap_df.to_string(index=False)

        return f"""
        Your GOAL: Write a story based on an XAI output to help everyday non-technical users 
        understand LLMs.

        CONTEXT:
        A large language model was used to {task_description}.

        The model was given the following instructions:
        "{system_prompt}"

        The model was then given the following patient profile to evaluate.
        This is the text that TokenSHAP analyzed, the Shapley values below
        refer exclusively to the tokens in this profile:
        "{original_prompt}"

        The model produced the following response:
        "{model_response}"

        To understand which parts of the prompt most influenced this response,
        a TokenSHAP analysis was performed. 

        HOW TOKENSHAP WORKS:
        TokenSHAP treats each token/word as a 'player' in a cooperative game. 
        It measures a token's 'marginal contribution', meaning, how much the model's response 
        changes when that specific token is added to various random subsets of the other 
        tokens. Each token (word) in the prompt was assigned an estimated Shapley value 
        representing its contribution to the model's output.

        INTERPRETING THE SHAPLEY VALUES:
        The values below are normalized to the range [-1, 1], centered on zero.
        A positive value means that when this token was present, the model's output was more 
        similar (using cosine similarity) to the final response, the closer to +1, the 
        stronger that influence. 
        A value near zero means the token had little to no effect on the output either way.
        A negative value means the token actively pulled the model's output away from the 
        final response when present, the closer to -1, the more destabilizing its influence.

        The TokenSHAP table below is sorted from most positively influential (top) to most 
        negatively influential (bottom):

        {shap_table}

        GUIDELINES:
        Before writing the narrative, first observe:
        - Which tokens sit at the extremes of the table (top and bottom) and how do 
        their values compare to the middle?
        - Are the values spread out or clustered closely together?
        - How important the patient's risk factors (age, hypertension, glucose level, etc.) 
        were.

        Use these observations to calibrate your narrative, 
        the top and bottom tokens are the most influential.

        TASK:
        Based on these estimated Shapley values, generate a plausible, fluent story 
        explaining WHY the model could have responded the way it did. 
        Focus on the tokens with the highest absolute estimated Shapley values. 
        Try to explain how the most influential tokens shaped the response 
        and why they were important, and note any potential interesting interactions 
        between them that fit the story. The main goal is to connect the dots between 
        the prompt, the influential tokens, and the model's response in a way that 
        makes sense, not to create a story that uses some of the (less influential) tokens 
        in a superficial way just for the sake of crafting a story.
        For example, if a patient's age did not have a high Shapley value, that would be 
        an interesting insight to weave into the narrative, as it might be surprising that 
        age was not a key driver of the model's response.
        If there are no outlying influential tokens, this is also an insight that can be 
        woven into the narrative.
        Your target audience is a non-technical user who wants to understand how the LLM 
        formed its response, so aim for an accessible and engaging explanation.
        There is no need to enumerate individual tokens outside of the story; 
        instead, weave the key insights into a coherent narrative that connects the prompt, 
        the influential tokens, and the model's response.
         
        Conclude with a short summary. Limit your answer to 8 sentences.
        """

    def generate_story(
        self,
        token_shap_instance,
        original_prompt:  str,
        system_prompt:    str,
        task_description: str,
        temperature:      float,
    ) -> tuple[str, str]:
        shap_df  = self.build_shap_df(token_shap_instance)
        baseline = token_shap_instance.baseline_text

        narrative_prompt = self.generate_narrative_prompt(
            original_prompt=original_prompt,
            system_prompt=system_prompt,
            model_response=baseline,
            shap_df=shap_df,
            task_description=task_description,
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": narrative_prompt}],
            temperature=temperature,
            max_tokens=1000,
        )

        return response.choices[0].message.content, narrative_prompt
