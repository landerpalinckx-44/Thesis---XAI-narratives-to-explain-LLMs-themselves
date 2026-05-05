# =============================================================================
# prompts.py — Prompt templates for the MExGen German credit experiment
# =============================================================================


def build_gc_prompt(applicant_profile: dict) -> tuple[str, list[str], list[str], list[bool]]:
    """
    Builds the German credit prompt in MExGen list format for a single applicant.

    Returns a tuple of:
    - system_prompt : frozen instructions passed to the model as a system message.
                      MExGen never perturbs this.
    - prompt        : list of prompt elements. Each feature is its own element so
                      MExGen can attribute to it individually.
    - unit_types    : parallel list marking each element as "p" (to be attributed)
                      or "n" (not of interest / never perturbed).
    - ind_segment   : parallel bool list marking which elements to segment into
                      words. True for all feature lines, False for structural text.

    Parameters
    ----------
    applicant_profile : dict
        A dict of {feature_name: value} for a single row from df_final_readable.
    """

    system_prompt = (
        "You are a credit scoring model. Your task is to predict whether a loan applicant "
        "is a good or bad credit risk based on their profile. "
        "Respond with only 'Good' or 'Bad'.\n\n"
        "Feature reference:\n"
        "- Duration: loan duration in months (dataset range: 4-72 months)\n"
        "- Credit amount: loan amount in Deutsche Mark (dataset range: 250-18424 DM)\n"
        "- Age: age in years (dataset range: 19-75 years)\n"
        "- Credit history: critical account, delay in past, existing credits paid back duly, "
        "all credits at this bank paid back duly, no credits/all paid back duly\n"
        "- Savings account/bonds: < 100 DM, 100-500 DM, 500-1000 DM, \u2265 1000 DM, "
        "unknown/no savings account\n"
        "- Job: unemployed/unskilled non-resident, unskilled resident, "
        "skilled employee/official, management/self-employed/highly qualified"
    )

    # Build feature lines from the profile dict
    feature_lines = [f"{col}: {val}\n" for col, val in applicant_profile.items()]

    # Structural elements (not attributed to)
    prefix   = "Applicant profile:\n"

    # Prompt is a LIST — this is critical for QA with MExGen.
    # Elements marked "n" (not of interest) won't be attributed to.
    # Only the feature lines will be segmented and attributed.
    prompt = [prefix] + feature_lines

    unit_types = (
        ["n"]                        # prefix
        + ["p"] * len(feature_lines) # one "p" per feature
    )

    ind_segment = (
        [False]                      # prefix — don't segment
        + [True] * len(feature_lines)# segment each feature into words
    )

    return system_prompt, prompt, unit_types, ind_segment
