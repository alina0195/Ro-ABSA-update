import re
import wandb
import os
import yaml
import pandas as pd
from pathlib import Path
from huggingface_hub import login
login(token="xxx")
import chat
import config
import json, re
import time

class Config:
    WANDB_INIT_NAME = "llama3_translation"
    ROOT = os.getcwd()
    DATASET_RO_PATH =  ROOT + os.sep +'data/new_data/absa/df_with_only_rare_lbls.csv'
    DATASET_SAVE_PATH =  ROOT + os.sep +'data/new_data/absa/df_train-original_and_translated2.csv'
    config_path = str(Path(__file__).parent) + os.sep + 'translate-llama.yaml'

with open(Config.config_path, "r") as f:
    prompt_config = yaml.safe_load(f)

prompt_config = config.ChatCompletionConfig(**prompt_config)

MODEL_NAME = prompt_config.model
chat_context = chat.ChatContextCreator(config=prompt_config)

run = wandb.init(project="roabsa_translate", 
                 name=Config.WANDB_INIT_NAME)


def remove_prompt(text):
    prompt = "Extract pairs of aspect categories with their corresponding opinions from the following Romanian review:"
    if prompt in text:
        text = text.replace(prompt,'')
    return text.strip()

df = pd.read_csv(Config.DATASET_RO_PATH)

df['absa_input'] = df['absa_input'].apply(remove_prompt)


def format_prompt(text, src_lang, tgt_lang):
    PROMPT_TEMPLATE  = chat_context.user_prompt.format(text=text, src_lang=src_lang, tgt_lang=tgt_lang)
    chat_context.clear_messages()
    
    chat_context.add_messages(user_content = PROMPT_TEMPLATE)
    
    return chat_context.messages


def prepare_df(df, src_lang, tgt_lang, col_name):
    df['translate_prompt']=df.apply(lambda x: format_prompt(x[col_name], 
                            src_lang=src_lang, tgt_lang=tgt_lang),
                        axis=1)
    print(f'Done formatting the input for {src_lang}->{tgt_lang}')

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
            "translated_text": response_text,
            "language": ""
        }


def infer_1_chain(df, src_lang, tgt_lang, col_name):
    df = prepare_df(df, src_lang, tgt_lang, col_name)
    print('DF after preparation:', df)
    
    results=[]
    languages = []
    for idx, row in df.iterrows():
        prediction = chat_context.chat_completion(messages=row['translate_prompt'])
        decoded_prediction = parse_model_output(prediction)
        results.append(decoded_prediction['translated_text'])
        languages.append(decoded_prediction['language'])
        
        if idx%10==0:
            time.sleep(5)
            
    df[f'translated_{src_lang}_{tgt_lang}'] = results
    df['language'] = languages
    
    df.drop(columns=['translate_prompt'], inplace=True)
    df.to_csv(Config.DATASET_SAVE_PATH, index=False)
    return df    
 
 
df = infer_1_chain(df=df, src_lang='Romanian', tgt_lang='English', col_name='absa_input')
df_table = wandb.Table(columns=['absa_input', 'absa_target', 'translated_Romanian_English', 'language'], dataframe=df)
run.log({f"df_ro_en": df_table})

df = infer_1_chain(df=df, src_lang='English', tgt_lang='Chinese', col_name='translated_Romanian_English')
df_table = wandb.Table(columns=['absa_input', 'absa_target', 'translated_English_Chinese', 'language'], dataframe=df)
run.log({f"df_en_chz": df_table})

df = infer_1_chain(df=df, src_lang='Chinese', tgt_lang='Romanian', col_name='translated_English_Chinese')
df_table = wandb.Table(columns=['absa_input', 'absa_target', 'translated_Chinese_Romanian', 'language'], dataframe=df)
run.log({f"df_chz2_ro": df_table})