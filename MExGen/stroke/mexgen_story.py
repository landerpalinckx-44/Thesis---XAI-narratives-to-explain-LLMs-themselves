# =============================================================================
# mexgen_story.py — MExGenStory class for stroke narrative generation
# =============================================================================

import pandas as pd
from openai import OpenAI


class MExGenStory:
    """
    Generates narratives explaining MExGen C-LIME attribution results using an LLM.
    Analogous to TokenSHAPstory but for MExGen word-level attributions.
    """

    def __init__(self, api_key: str, model_name: str, base_url: str = None):
        self.client     = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def build_attribution_df(self, output_dict_words: dict, scalarizer: str) -> pd.DataFrame:
        """
        Converts MExGen word-level attribution results into a clean DataFrame,
        filtering out non-attributed ("n") units.
        """
        df = pd.DataFrame(output_dict_words["attributions"])
        return (
            df[df["unit_types"] != "n"][["units", scalarizer]]
            .sort_values(by=scalarizer, ascending=False)
            .rename(columns={"units": "token", scalarizer: "attribution_score"})
            .reset_index(drop=True)
        )

    def generate_narrative_prompt(
        self,
        system_prompt:    str,
        original_prompt:  str,
        model_response:   str,
        attribution_df:   pd.DataFrame,
        task_description: str,
    ) -> str:
        attribution_table = attribution_df.to_string(index=False)

        return f"""
        Your GOAL: Write a story based on an XAI output to help everyday non-technical users 
        understand LLMs.

        CONTEXT:
        A large language model was used to {task_description}.

        The model was given the following instructions:
        "{system_prompt}"

        The model was then given the following patient profile to evaluate.
        This is the text that MExGen analyzed, the attribution scores below
        refer exclusively to the words in this profile:
        "{original_prompt}"

        The model produced the following response:
        "{model_response}"

        To understand which parts of the prompt most influenced this response,
        a MExGen C-LIME (Multi-Level Explanations for Generative Language Models,
        Constrained LIME variant) analysis was performed.

        HOW C-LIME WORKS:
        C-LIME trains a simple linear model to locally approximate the LLM's behavior
        by systematically removing specific words and measuring how much the model's
        response changes as a result (using BERTScore semantic similarity). The final
        attribution scores are the mathematical weights derived from this model,
        quantifying exactly how much influence each word had on the model's final answer.

        INTERPRETING THE SCORES:
        The values below are normalized to the range [-1, 1], centered on zero.
        A positive score means: when this word was present, the model's response
        was more similar (using BERTScore semantic similarity) to the final response,
        the closer to +1, the stronger that influence.
        A value near zero means: the word had little to no effect on the output
        either way.
        A negative score means: this word's presence pulled the model's output
        away from the final response when present, the closer to -1, the more
        destabilizing its influence.

        The attribution table below is sorted from most positively influential (top)
        to most negatively influential (bottom):

        {attribution_table}

        GUIDELINES:
        Before writing the narrative, first observe:
        - Which words sit at the extremes of the table (top and bottom) and how do
          their values compare to the middle?
        - Are the values spread out or clustered closely together?
        - How important the patient's risk factors (age, hypertension, glucose level, etc.) 
        were.

        Use these observations to calibrate your narrative, 
        the top and bottom words are the most influential.

        TASK:
        Based on these C-LIME attribution scores, generate a plausible, fluent story
        explaining WHY the model could have responded the way it did. 
        Focus on the words with the highest absolute attribution scores. 
        Try to explain how the most influential words shaped the response 
        and why they were important, and note any potential interesting interactions 
        between them that fit the story. The main goal is to connect the dots between 
        the prompt, the influential words, and the model's response in a way that 
        makes sense, not to create a story that uses some of the (less influential) words 
        in a superficial way just for the sake of crafting a story.
        For example, if a patient's age did not have a high attribution score, that would 
        be an interesting insight to weave into the narrative, as it might be surprising 
        that age was not a key driver of the model's response.
        If there are no outlying influential words, this is also an insight that can be 
        woven into the narrative. 
        Your target audience is a non-technical user who wants to understand how the
        LLM formed its response, so aim for an accessible and engaging explanation.
        There is no need to enumerate individual words outside of the story; 
        instead, weave the key insights into a coherent narrative that connects the prompt,
        the influential words, and the model's response.
         
        Conclude with a short summary. Limit your answer to 8 sentences.
        """

    def generate_story(
        self,
        output_dict_words: dict,
        scalarizer:        str,
        system_prompt:     str,
        original_prompt:   str,
        task_description:  str,
        temperature:       float,
    ) -> tuple[str, str]:
        attribution_df = self.build_attribution_df(output_dict_words, scalarizer)
        model_response = output_dict_words["output_orig"].output_text[0]

        narrative_prompt = self.generate_narrative_prompt(
            system_prompt=system_prompt,
            original_prompt=original_prompt,
            model_response=model_response,
            attribution_df=attribution_df,
            task_description=task_description,
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": narrative_prompt}],
            temperature=temperature,
            max_tokens=1000,
        )

        return response.choices[0].message.content, narrative_prompt
