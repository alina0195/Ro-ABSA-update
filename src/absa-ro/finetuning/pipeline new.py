import re
import datetime
import string
import wandb
import torch
import string
import evaluate
import random
import nltk
import os
import torch, gc
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import (BitsAndBytesConfig, AutoModelForSeq2SeqLM,
                          TrainingArguments,T5ForConditionalGeneration, 
                          )
       
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from huggingface_hub import login
login(token="xxx")

nltk.download('punkt')
torch.set_warn_always(True)

nltk.download('punkt')
torch.set_warn_always(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse

parser = argparse.ArgumentParser(description="Script configuration")
parser.add_argument("--version", type=int, default=1, help="Version for model's name saved")
args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class config:
  HF_TOKEN='xxx'
  CATEGORIES = [
    "product","shop diversity","staff competency","shop organization","service",
    "quality","price","environment","staff availability","misc","delivery",
    "promotions","tech support","return warranty","security"
        ]
  LABELS = ["positive","negative","neutral"]

  SEED = 42
  ROOT = os.getcwd()
  
  DATASET = ROOT + os.sep +'data/new_data/absa/test_absaPairs.csv'

  MODEL_ALSA = ROOT+ '/models/new_models/alsa_RoBERT-large_v2.pt'
  TOKENIZER_ALSA = ROOT+ '/models/new_models/alsa_RoBERT-large_v2.pt'
  MODEL_ATE = ROOT+ '/models/new_models/atc_RoBERT-large_encdec_v2.pt'
  TOKENIZER_ATE = ROOT+ '/models/new_models/atc_RoBERT-large_encdec_v2.pt'

  MAX_SOURCE_LEN = 512
  MAX_TARGET_LEN = 30
  THRESHOLD_ATEC = 0.7
  BATCH_SIZE = 4
  DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


"""
Evaluate ABSA pipeline: review -> ATEC -> ALSA
Input: review
Output: ATC_1 is polarity; ATC_2 is polarity; ... ; ATC_n is polarity
"""
# absa_target:  << promotions is negative; shop diversity is negative >>
# text,id,aspects_polarities,annotator,data_origin,text_cleaned,absa_input,absa_target

set_seed(config.SEED)

wandb.init(project="new_pipeline_absa", name=f"V{args.version}",
             config={
               'model_ATC': config.MODEL_ATE,
               'model_ALSC': config.MODEL_ALSA,
             }) 


def format_text_for_df_atec(df, text_col_name, new_text_col_name):
    df[new_text_col_name] = df[text_col_name].apply(lambda x: x.lower())
    return df

def f1(pred, target):
      return f1_score(target, pred, average='weighted')

def recall(pred, target):
    pred = pred.split(';')
    target = target.split(';')
    
    pred = [p.strip() for p in pred]
    target = [p.strip() for p in target]

    sum = 0
    already_seen = []
    for p in pred:
        if p in target and p not in already_seen:
            sum += 1
            already_seen.append(p)
    sum=sum/(len(target))
    return sum

def precision(pred, target):
    pred = pred.split(';')
    target = target.split(';')
    
    pred = [p.strip() for p in pred]
    target = [p.strip() for p in target]

    correct = 0
    already_seen = []
    for p in pred:
        if p in target and p not in already_seen:
            correct += 1
            already_seen.append(p)
    
    return correct / len(pred) if len(pred) > 0 else 0

def label_f1(precision, recall):
    if precision + recall == 0:
        return 0.0  
    return 2 * (precision * recall) / (precision + recall)
  
def load_model(name, token, labels):
    model = AutoModelForSequenceClassification.from_pretrained(
                name,
                token=token,
                num_labels=len(labels))
    # model.load_state_dict(torch.load(name, map_location=config.DEVICE))
    model.to(config.DEVICE)
    
    tokenizer = AutoTokenizer.from_pretrained(name,
                                              use_fast=False,
                                              token=token)
    return model, tokenizer

model_atec, tokenizer_atec= load_model(name=config.MODEL_ATE, token=config.HF_TOKEN, labels = config.CATEGORIES)
print('Loaded ATEC model')

model_alsa,tokenizer_alsa = load_model(name=config.MODEL_ALSA, token=config.HF_TOKEN, labels = config.LABELS)
print('Loaded ALSA model')

model_alsa.eval()
model_atec.eval()


df = pd.read_csv(config.DATASET)
df = format_text_for_df_atec(df, 'text_cleaned','text_cleaned_atec')
df.dropna(subset=['text_cleaned_atec'],inplace=True)

# df = df[:8]

def sigmoid(x): return 1 / (1 + np.exp(-x))

label2id = {label: i for i, label in enumerate(config.LABELS)}
id2label = {i: label for label, i in label2id.items()}

def predict_categories(text):
    """Return list of predicted aspect categories"""
    inputs = tokenizer_atec(text, truncation=True, max_length=config.MAX_SOURCE_LEN, padding=True, return_tensors="pt").to(config.DEVICE)
    with torch.no_grad():
        logits = model_atec(**inputs).logits
        probs = sigmoid(logits.cpu().numpy())
        preds = (probs >= config.THRESHOLD_ATEC).astype(int)
    cats = []
    for row in preds:
        cats.append([config.CATEGORIES[i] for i, v in enumerate(row) if v == 1])
    return cats[0]

def predict_sentiment(text, aspect):
    """Return predicted sentiment for given (text, aspect) pair"""
    inp = f"{text}[SEP]{aspect}"
    encoded = tokenizer_alsa(inp, truncation=True, max_length=config.MAX_SOURCE_LEN, padding=True, return_tensors="pt").to(config.DEVICE)
    with torch.no_grad():
        logits = model_alsa(**encoded).logits
        pred = torch.argmax(logits, dim=1).item()
    return config.LABELS[pred]

print("Running EVALUATION...")

def parse_gold(s):
        if not isinstance(s, str): return []
        pairs = []
        for p in s.split(";"):
            p=p.strip()
            if not p:
                continue
            if " is " in p:
                parts = re.split(r"\s+is\s+", p, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 2:
                    aspect, polarity = parts
                    pairs.append((aspect.strip(), polarity.strip()))
        return pairs

texts = df['text_cleaned_atec'].astype(str).tolist()
gold_pairs = df["absa_target"].apply(parse_gold).tolist()
print('GOLD PAIRS:', gold_pairs)
pred_triplets, gold_triplets = [], []
    
for i, text in tqdm(enumerate(texts), total=len(texts)):
    pred_cats = predict_categories(text)
    pred_pairs = [(c, predict_sentiment(text, c)) for c in pred_cats]
    pred_triplets.append(pred_pairs)
    gold_triplets.append(gold_pairs[i])

print('GOLD TRIPLETS:', gold_triplets)
print('pred TRIPLETS:', pred_triplets)

# Flatten predictions
y_true, y_pred = [], []
for gold, pred in zip(gold_triplets, pred_triplets):
    gold_set = set(gold)
    pred_set = set(pred)
    for p in pred_set.union(gold_set):
        y_true.append(1 if p in gold_set else 0)
        y_pred.append(1 if p in pred_set else 0)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="binary"
)

print("\n================ E2E ABSA RESULTS ================")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")
print("===============================================\n")

# Optional: display sample predictions
for i in range(5):
    print(f"Text: {texts[i][:150]}...")
    print("Predicted:", pred_triplets[i])
    print("Gold:", gold_triplets[i])
    print()