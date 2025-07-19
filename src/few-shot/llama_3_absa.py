import pandas as pd
def retrieve_llama_predictions(few_shot_df: pd.DataFrame
			   , test_df: pd.DataFrame
			   , model: str = "llama3.3:latest") -> pd.DataFrame:

    df_merged = few_shot_df.merge(test_df, on='text', how='outer', indicator=True)
    
    true_labels_list = []
    pred_labels_list = []
    
    for index, row in df_merged.iterrows():
        data = {
          "model": model,
          "messages": [
                    {
                      "role": "system",
                      "content": row['few_shot_prompt_gpt'],
                    },
                    {"role": "user", "content": f"review: {row['text']}"},
            ],
          "temperature": 0.0
          }
    
    
        response = requests.post(url=url,
                               headers=headers,
                               data=json.dumps(data))
    
        if response:
          response_text = json.loads(response.text)['choices'][0]['message']['content'].lower()
    
        true_labels = row['absa_target'].split("; ")
        pred_labels = response_text.split("; ")
        
        true_labels_list.append(true_labels)
        pred_labels_list.append(pred_labels)
