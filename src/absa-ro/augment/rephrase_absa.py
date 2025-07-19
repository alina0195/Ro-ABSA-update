import re
import wandb
import torch
import os
import yaml
import pandas as pd
from sklearn.metrics import f1_score
from pathlib import Path
from huggingface_hub import login
login(token="xxx")
import chat
import config
import json, re
import time

class Config:
    WANDB_INIT_NAME = "llama3_rephrase"
    ROOT = os.getcwd()
    DATASET_RO_PATH =  ROOT + os.sep +'data/new_data/absa/df_with_only_rare_lbls.csv'
    DATASET_SAVE_PATH =  ROOT + os.sep +'data/new_data/absa/df_train-original_and_rephrased.csv'
    config_path = str(Path(__file__).parent) + os.sep + 'cot-llama.yaml'

with open(Config.config_path, "r") as f:
    prompt_config = yaml.safe_load(f)

prompt_config = config.ChatCompletionConfig(**prompt_config)

MODEL_NAME = prompt_config.model
chat_context = chat.ChatContextCreator(config=prompt_config)

run = wandb.init(project="roabsa_rephrasing", 
                 name=Config.WANDB_INIT_NAME)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        return {
            'input_ids' : item['input_ids'],
            'attention_mask' : item['attention_mask']
        }
   
df = pd.read_csv(Config.DATASET_RO_PATH)

def format_prompt(text, target):
    PROMPT_TEMPLATE  = chat_context.user_prompt.format(text=text, target=target)
    chat_context.clear_messages()
    
    chat_context.add_messages(user_content = PROMPT_TEMPLATE)
    
    return chat_context.messages


def prepare_df(df):
    df['rephrasing_prompt']=df.apply(lambda x: format_prompt(x['absa_input'], 
                            x['absa_target'] 
                            ),
                        axis=1)
    print('Done formatting the input')

    return df


def parse_model_output(response_text):
    try:
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            clean_json = json_match.group(1)
        else:
            clean_json = response_text.strip()

        return json.loads(clean_json)

    except json.JSONDecodeError as e:
        print(f"JSON parse failed: {e}")
        return {
            "rephrased_review": response_text,
            "label": ""
        }


def infer(df):
    df = prepare_df(df)
    print('DF after preparation:', df)
    
    results=[]
    new_labels = []
    for idx, row in df.iterrows():
        prediction = chat_context.chat_completion(messages=row['rephrasing_prompt'])
        decoded_prediction = parse_model_output(prediction)
        results.append(decoded_prediction['rephrased_review'])
        new_labels.append(decoded_prediction['label'])

        if idx%10==0:
            time.sleep(5)
            
    df['text_rephrased'] = results
    df['new_label'] = new_labels
    
    df.drop(columns=['rephrasing_prompt'], inplace=True)
    df.to_csv(Config.DATASET_SAVE_PATH, index=False)
    return df    
 
 
df = infer(df=df)
df_table = wandb.Table(columns=['absa_input','absa_target','text_rephrased','new_label'], dataframe=df)
run.log({f"df": df_table})