import pandas as pd
from typing import List
import re
from openai import OpenAI

def retrieve_gpt_predictions(few_shot_df: pd.DataFrame
			   , test_df: pd.DataFrame
			   , model: str = "gpt-4o") -> pd.DataFrame:

    df_merged = few_shot_df.merge(test_df, on='text', how='outer', indicator=True)
    
    true_labels_list = []
    pred_labels_list = []
    
    for index, row in df_merged.iterrows():
        response = client.chat.completions.create(
          model=model,
          messages=[
            {
              "role": "system",
              "content": row['few_shot_prompt_gpt']
            },
            {
                "role": "user",
                "content": f"review: {row['text']}"
            }
          ]
        )
    
        output = str(response.choices[0].message).replace("\\n", " ").replace("\n", " ")
    
        pattern = r"([a-zA-Z\s]+):(\w+)"
        matches = re.findall(pattern, outputs)
        labels = ''
        for category, polarity in matches:
            labels += f"{category.strip().upper()}:{polarity.strip().upper()}, "
        labels = labels[:-2]
        test_df.at[index, 'generated_aspects_with_polarities'] = labels
        true_labels = row['absa_target'].strip().upper()
    
        true_labels = true_labels.split(', ')
        pred_labels = labels.split(', ')
    
        true_labels.sort()
        pred_labels.sort()
    
        true_labels_list.append(true_labels)
        pred_labels_list.append(pred_labels)