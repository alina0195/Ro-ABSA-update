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
                "content": f"review: {row['text']} aspect: {row['category']}"
            }
          ]
        )
        
        output = str(response.choices[0].message)
        match = re.search(r"content='(.*?)'", output)
        if match:
            match_group = match.group(1).lower()
            pred_polarity = 'positive' if 'positive' in match_group else 'negative' if 'negative' in match_group else 'neutral'
        true_polarity = row['polarity'].lower()
    
        true_labels_list.append(true_polarity)
        pred_labels_list.append(pred_polarity)