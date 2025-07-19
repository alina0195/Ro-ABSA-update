import pandas as pd
import chat
import config
import time, os, json, re
from pathlib import Path
import yaml
import wandb
import torch

class Config:
    WANDB_INIT_NAME = "llama3_quad2text_fewshot"
    ROOT = os.getcwd()
    DATASET_RO_PATH =  ROOT + os.sep +'data/new_data/absa/df_with_only_rare_lbls.csv'
    DATASET_SAVE_PATH = 'train_absaPairs_quad2text_2.csv'
    SAVE_EVERY = 5                
    RETRY_PAUSE = 0                 

    config_path = str(Path(__file__).parent) + os.sep + 'quad2text_llama.yaml'

with open(Config.config_path, "r") as f:
    prompt_config = yaml.safe_load(f)

prompt_config = config.ChatCompletionConfig(**prompt_config)
MODEL_NAME = prompt_config.model
chat_context = chat.ChatContextCreator(config=prompt_config)

run = wandb.init(project="roabsa_quad2text",
                 name=Config.WANDB_INIT_NAME)

full_df = pd.read_csv(Config.DATASET_RO_PATH)

full_df = full_df[345:]

if os.path.isfile(Config.DATASET_SAVE_PATH):
    done_df = pd.read_csv(Config.DATASET_SAVE_PATH)
    processed_ids = set(done_df.index)          
    print(f"{len(done_df)} already processed.")
else:
    done_df = pd.DataFrame()
    processed_ids = set()

df = full_df.loc[~full_df.index.isin(processed_ids)].copy()
print(" Processed:", len(df))

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        return {"input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"]}

def format_prompt(target):
    PROMPT_TEMPLATE = chat_context.user_prompt.format(target=target)
    chat_context.clear_messages()
    chat_context.add_messages(user_content=PROMPT_TEMPLATE)
    return chat_context.messages

def prepare_df(df):
    df["quad2text_prompt"] = df.apply(
        lambda x: format_prompt(x["absa_target"]), axis=1
    )
    return df

def parse_model_output(response_text):
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.S)
        clean_json = match.group(1) if match else response_text.strip()
        return json.loads(clean_json)
    except json.JSONDecodeError as e:
        print("JSON parse failed:", e)
        return {"review": response_text, "echo_targets": ""}

def infer(df):
    df = prepare_df(df)
    results, new_labels = [], []

    for i, (idx, row) in enumerate(df.iterrows(), 1):
        tries = 0
        while True:
            try:
                raw_pred = chat_context.chat_completion(messages=row["quad2text_prompt"])
                break
            except Exception as e:
                tries += 1
                print(f"API Error  (tried {tries}): {e}")
                time.sleep(Config.RETRY_PAUSE)

        decoded = parse_model_output(raw_pred)
        results.append(decoded.get("review", raw_pred))
        new_labels.append(decoded.get("echo_targets", ""))

        df.loc[idx, "review_generated"] = results[-1]
        df.loc[idx, "echo_targets"] = new_labels[-1]

        if i % Config.SAVE_EVERY == 0 or i == len(df):
            combined = pd.concat([done_df, df.loc[:idx]], axis=0)
            combined = combined.sort_index()               
            combined.drop(columns=["quad2text_prompt"], errors="ignore", inplace=True)
            combined.to_csv(Config.DATASET_SAVE_PATH, index=False)
            print(f" Saved checkpoint after {i} / {len(df)} rows")

    df.drop(columns=["quad2text_prompt"], inplace=True, errors="ignore")
    final_df = pd.concat([done_df, df], axis=0).sort_index()
    return final_df

final_df = infer(df)
wandb_table = wandb.Table(
    columns=["absa_input", "absa_target", "review_generated", "echo_targets"],
    dataframe=final_df,
)
run.log({"df": wandb_table})
print("Done")