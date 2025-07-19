from openai import OpenAI, AsyncOpenAI
import os
import time
import pandas as pd
import json
import re

os.environ['OPENAI_API_KEY'] = 'sk-proj-xxxx'
client = OpenAI(api_key=os.environ['OPENAI_API_KEY']) 

def ask_chatgpt(system_prompt, user_prompt,
                model="gpt-4o", temperature = 0.7, 
                max_retries=3):

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries} failed with error: {e}")
            time.sleep(2)  # basic retry delay

    return "ERROR: API call failed after multiple attempts."


def check_entailment(text, label):
    system_prompt = "You are a helpful assistant specialized in evaluating aspect-based sentiment consistency."
    user_prompt = (
        f"Evaluate whether the original list with category and the opinion related to it still apply to the following augmented review.\n\n"
        f"Augmented Review: {text}\n"
        f"Original target: {label}\n\n"
        f"The target entails with the review only if all of the category-opinion pairs chained by <;> are correct. Respond with ONLY one tag: ENTAILED or NOT_ENTAILED."
    )

    result = ask_chatgpt(system_prompt, user_prompt)

    if "not_entailed" in result.lower():
        return "not_entailed"
    elif "entailed" in result.lower():
        return "entailed"
    else:
        return "error"


def qc_decision(raw_response,
                min_coherence=3):

    match = re.search(r'\{.*\}', raw_response, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in model output.")
    data = json.loads(match.group(0))

    keep = (
        data.get("coherence_score", 0) >= min_coherence and
        data.get("meaning_consistency", False) is True
    )
    return keep


def check_qc(text_aug, label):
    system_prompt = "You are a quality-control assistant for data augmentation in Aspect-Based Sentiment Analysis."

    user_prompt = ''' TASK  
            Given:
            - a list of category-sentiment pairs mentioned in a customer review  
            - its AUGMENTED_REVIEW (generated automatically)
            Return a JSON object with the following keys:
            {
            "coherence_score":  <int>,    # 1–5 (1 = gibberish, 5 = perfectly fluent/grammatical)
            "meaning_consistency": <bool> # true if sentiment toward TARGET from all pairs is unchanged
            }
            GUIDELINES  
            1. **Coherence**: penalise if the augmented text is ungrammatical, contradictory, or incomplete.  
            2. **Meaning consistency**: the overall stance (positive/negative/neutral) toward TARGET must match the original in all pairs from label.  

            INPUT  
            LABEL: {{label}}  
            AUGMENTED_REVIEW: {{text_aug}}

            OUTPUT  
            Respond with one JSON object only. Do **not** add extra keys, text, or markdown.
   '''

    result = ask_chatgpt(system_prompt, user_prompt)
    return result


# def add_entailment_column(df):
#     entailments = []
#     qualities = []
#     passes = []
    
#     for idx, row in df.iterrows():
#         if row["data_origin"] != "manual":
#             print(f"Processing row {idx}...")
#             entailment_result = check_entailment(row["absa_input"], row["absa_target"])
#             entailments.append(entailment_result)
            
#             if entailment_result == 'entailed':
#                 qc_result = check_qc(row['absa_input'], row['absa_target'])
#                 qualities.append(qc_result)
#                 rule_passed = qc_decision(raw_response=qc_result, min_coherence=2)
#                 passes.append(rule_passed)
#             else:
#                 qualities.append('none')
#                 passes.append('not_keep')
                
#             time.sleep(1)  # Avoid rate limits
#         else:
#             entailments.append('original')
#             qualities.append('original')
#             passes.append('keep')
            
#     df["entailment"] = entailments
#     df["qualities"] = qualities
#     df["rule_passed"] = passes
    
#     return df


def add_entailment_column(df,
                          checkpoint_path="absa_train_with_enitalments_for_aug2.csv",
                          save_every=25):                # rows per checkpoint
    if os.path.exists(checkpoint_path):
        print(" Resuming from checkpoint …")
        df = pd.read_csv(checkpoint_path)

    for col in ["entailment", "qualities", "rule_passed"]:
        if col not in df:
            df[col] = pd.NA                         # fill with <NA> (missing)

    for i, row in df.iterrows():
        if pd.notna(df.at[i, "rule_passed"]):       # already processed, skip
            continue
        try:
            if row["data_origin"] != "manual":
                entail = check_entailment(row["absa_input"], row["absa_target"])
                df.at[i, "entailment"] = entail

                if entail == "entailed":
                    qc_raw = check_qc(row["absa_input"], row["absa_target"])
                    keep   = qc_decision(raw_response=qc_raw,
                                        min_coherence=2)
                    df.at[i, "qualities"] = qc_raw
                    df.at[i, "rule_passed"] = "keep" if keep else "not_keep"
                else:
                    df.at[i, ["qualities", "rule_passed"]] = ["none", "not_keep"]
            else:
                df.at[i, ["entailment", "qualities", "rule_passed"]] = \
                    ["original", "original", "keep"]

        except Exception as e:              # log the error, keep going
            print(f" Row {i} failed: {e}")
            df.at[i, ["entailment", "qualities", "rule_passed"] ] = ["error","error","error"]

        if i % save_every == 0:
            df.to_csv(checkpoint_path, index=False)
            print(f" Saved checkpoint at row {i}")

        time.sleep(1)  # avoid rate limits

    df.to_csv(checkpoint_path,  index=False)
    print("✅  Finished & saved final checkpoint")
    return df


if __name__ == "__main__":
    df = pd.read_csv("./train_absaPairs_aug_v2.csv")
    # df = df[df['data_origin']!='manual'][:4]
    df = add_entailment_column(df)
