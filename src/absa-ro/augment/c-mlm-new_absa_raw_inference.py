import pandas as pd
import os
import torch
from transformers import AutoTokenizer, BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoModelForMaskedLM
from torch.utils.data import Dataset
import spacy
import wandb
import random
import argparse
import numpy as np

torch.set_warn_always(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description="Script configuration")
parser.add_argument("--model", type=str, default='', help="Pretrained model name")
parser.add_argument("--tokenizer", type=str, default='', help="Pretrained tokenizer name")
parser.add_argument("--local", type=bool, default=False, help="Is a local model")

args = parser.parse_args()
print(f'Task Running for\nmodel: {args.model}\ntokenizer: {args.tokenizer}\nlocal:{args.local}')


nlp = spacy.load("./ro_core_news_sm-3.8.0")
print('Spacy model loaded')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class config:
    SEED = 42
    ROOT = os.getcwd()
    MLM_PROBABILITY = 0.10
    DATASET_RO_PATH =  ROOT + os.sep +'data/new_data/absa/df_with_only_rare_lbls.csv'
    DATASET_SAVE_PATH =  ROOT + os.sep +'data/new_data/absa/df_train-augmented_and_cmlm_raw_v2.csv'
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if args.local==True:
        print('Model chosen from local repo')
        MODEL = ROOT + os.sep + 'models/new_models/' + args.model
        WANDB_INIT_NAME = args.model + ' - finetuned' 
    else:
        print('Model chosen from HF-HUB')
        MODEL = args.model 
        WANDB_INIT_NAME = MODEL

    TOKENIZER = args.tokenizer
    

set_seed(config.SEED)
df = pd.read_csv(config.DATASET_RO_PATH)

wandb.init(project="roabsa_cmlm_raw", name=config.WANDB_INIT_NAME,
           config={
                    "Condition": "meaningful POS: NOUN, VERB, ADJ",
                    "Description": "Augment ABSA pairs with MLM predictions",
                    "Mask rate": config.MLM_PROBABILITY,
                })


"""
In Conditional MLM, we mask tokens in the original review and condition the model on additional information like:
- The token which is masked should be meaningful - "NOUN", "VERB", "ADJ"
This allows the model to fill in the blanks in a way that is aware of the aspect context.
"""

def mask_meaningful_tokens(text, percent=0.3, min_masks=3):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    
    if len(text.split(' ')) < 20:
        percent = percent / 2
        min_masks =int(min_masks / 2 + 1) 
        
    candidate_indices = [
        i for i, token in enumerate(doc)
        if token.pos_ in {"NOUN", "VERB", "ADJ"} and not token.is_stop and token.is_alpha
    ]

    if not candidate_indices:
        return text  # fallback: don't mask if no good candidates

    num_tokens_to_mask = max(min_masks, int(len(tokens) * percent))
    num_tokens_to_mask = min(num_tokens_to_mask, len(candidate_indices))

    selected_indices = torch.randperm(len(candidate_indices))[:num_tokens_to_mask]
    selected_token_indices = [candidate_indices[i] for i in selected_indices]

    for idx in selected_token_indices:
        tokens[idx] = "[MASK]"

    return " ".join(tokens)


class ABSA_MLMDataset(Dataset):
    def __init__(self, tokenizer, reviews, targets):
        self.tokenizer = tokenizer
        self.inputs = []
        for review, target_str in zip(reviews, targets):
            combined = f"{review} [SEP] Aspects={target_str}"
            enc = tokenizer(combined, truncation=True, padding='max_length', max_length=512, return_tensors="pt")
            self.inputs.append(enc)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        item = {key: val.squeeze() for key, val in self.inputs[idx].items()}
        item['labels'] = item['input_ids'].clone()
        return item

model = AutoModelForMaskedLM.from_pretrained(config.MODEL)

tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER)

dataset = ABSA_MLMDataset(tokenizer, df['text_cleaned'], df['absa_target'])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                mlm=True,
                                                mlm_probability=config.MLM_PROBABILITY
                                                )


def batch_infer_after_training(model, tokenizer, texts, categories, device, mask_percent=config.MLM_PROBABILITY, max_length=512):
    from math import ceil
    model.eval()
    model.to(device)

    all_masked = []
    all_augmented = []
    batch_size = config.BATCH_SIZE
    num_batches = ceil(len(texts) / batch_size)

    for b in range(num_batches):
        batch_texts = texts[b * batch_size: (b + 1) * batch_size]
        batch_cats = categories[b * batch_size: (b + 1) * batch_size]

        input_ids_list = []
        attention_masks_list = []
        sep_indexes = []
        masked_reviews = []

        for text, cats in zip(batch_texts, batch_cats):
            masked = mask_meaningful_tokens(text, percent=mask_percent, min_masks=3)
            masked_reviews.append(masked) 
            conditioned_input = f"{masked} [SEP] Aspects={cats}"
            inputs = tokenizer(conditioned_input, truncation=True, padding="max_length",
                               max_length=max_length, return_tensors="pt")
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            sep_index = (input_ids == tokenizer.sep_token_id).nonzero(as_tuple=True)[1][0].item()

            input_ids_list.append(input_ids.squeeze())
            attention_masks_list.append(attention_mask.squeeze())
            sep_indexes.append(sep_index)

        input_ids_batch = torch.stack(input_ids_list).to(device)
        attn_mask_batch = torch.stack(attention_masks_list).to(device)

        labels = input_ids_batch.clone()
        for i, sep_index in enumerate(sep_indexes):
            labels[i, sep_index:] = -100
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids_batch, attention_mask=attn_mask_batch, labels=labels)
            predictions = torch.argmax(outputs.logits, dim=-1)

        for i, pred in enumerate(predictions):
            pred_ids = pred.tolist()
            pred_cut = pred_ids[:sep_indexes[i]]
            decoded = tokenizer.decode(pred_cut, skip_special_tokens=True)
            all_masked.append(masked_reviews[i])
            all_augmented.append(decoded)

    return all_masked, all_augmented


masked_reviews, augmented_reviews = batch_infer_after_training(
                                                                model=model,
                                                                tokenizer=tokenizer,
                                                                texts=df['text_cleaned'].tolist(),
                                                                categories=df['absa_target'].tolist(),
                                                                device=config.DEVICE,
                                                            )
table = wandb.Table(dataframe=pd.DataFrame({'initial_input':df['text_cleaned'], 
                                            "augmented_input": augmented_reviews, 
                                            "target": df["absa_target"], 
                                            "masked_input":masked_reviews})
                   )
wandb.log({"mlm_augmentation_table": table})

print('Masked rev:', masked_reviews)
print('Augmented rev:', augmented_reviews)

df["cmlm_raw_text"] = augmented_reviews
PROMPT_ABSA = 'Extract pairs of aspect categories with their corresponding opinions from the following Romanian review: '

df['cmlm_raw_text']=df['cmlm_raw_text'].apply(lambda x: PROMPT_ABSA + x + '</s>')
df.to_csv(config.DATASET_SAVE_PATH, index=False)