import pandas as pd


def extract_categories(text, categories):
	text_lower = text.lower()
	matched = []
	for category in categories:
		category_lower = category.lower()
		if category_lower in text_lower:
			matched.append(category_lower)
	return matched


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
    
        true_labels = row['all_categories'].split("; ")
        pred_labels = extract_categories(response_text, categories)
        
        true_labels_list.append(true_labels)
        pred_labels_list.append(pred_labels)
