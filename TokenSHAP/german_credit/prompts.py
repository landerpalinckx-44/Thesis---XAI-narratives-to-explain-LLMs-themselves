# =============================================================================
# prompts.py — Prompt templates for the German credit experiment
# =============================================================================

def build_gc_prompt(applicant_profile: str) -> tuple[str, str]:
    """
    Builds the German credit prompt for a single applicant instance.

    Returns a (system_prompt, user_prompt) tuple.
    - system_prompt: frozen instructions, feature reference, and formatting constraint. TokenSHAP never perturbs this.
    - user_prompt:   only the applicant profile. TokenSHAP splits and masks tokens exclusively from this string.

    Parameters
    ----------
    applicant_profile : str
        A string representation of a single row from df_final_readable.
    """

    system_prompt = """You are a credit scoring model. Your task is to predict whether a loan applicant is a good or bad credit risk based on their profile.
    Respond with only 'Good' or 'Bad'.

    Feature reference:
    - Duration: loan duration in months (dataset range: 4-72 months)
    - Credit amount: loan amount in Deutsche Mark (dataset range: 250-18424 DM)
    - Age: age in years (dataset range: 19-75 years)
    - Credit history: critical account, delay in past, existing credits paid back duly, all credits at this bank paid back duly, no credits/all paid back duly
    - Savings account/bonds: < 100 DM, 100-500 DM, 500-1000 DM, ≥ 1000 DM, unknown/no savings account
    - Job: unemployed/unskilled non-resident, unskilled resident, skilled employee/official, management/self-employed/highly qualified"""

    user_prompt = f"""Applicant profile:
    {applicant_profile}"""

    return system_prompt, user_prompt