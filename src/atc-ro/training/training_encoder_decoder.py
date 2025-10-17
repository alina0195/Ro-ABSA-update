import re
import wandb
import string
import evaluate
import random
import nltk
import os
import torch, gc
from torch import nn 
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoConfig, 
                          EarlyStoppingCallback,
                          AutoTokenizer)
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
         
from huggingface_hub import login
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import argparse
nltk.download('punkt')
torch.set_warn_always(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#model : "microsoft/deberta-v3-large", "answerdotai/ModernBERT-large", "readerbench/RoBERT-large"

parser = argparse.ArgumentParser(description="Script configuration")
parser.add_argument("--model", type=str, default='', help="Pretrained model name")
parser.add_argument("--batch", type=int, default=4, help="Batch size")
parser.add_argument("--acc_steps", type=int, default=2, help="Gradient accumulation steps")
parser.add_argument("--epochs", type=int, default=10, help="Number of Epochs")
parser.add_argument("--version", type=int, default=3, help="Version for model's name saved")
parser.add_argument("--description", type=str, default='', help="Will be placed in wandb run description")
parser.add_argument("--aug", type=str, default='', help="Augmentation type to include. Values: all/bt/c-mlm/rc/rephrasing/quad2text_trained/rephrasing+quad2text")
parser.add_argument("--train_path", type=str, default='train_absaPairs_aug_final.csv', help="Train csv base name")
parser.add_argument("--threshold", type=float, default=0.75, help="Threshold for probabilities")
parser.add_argument("--scheduler", type=str, default='constant', help="Type of scheduler")

args = parser.parse_args()
print(f'Task Running for model: {args.model}, batch size: {args.batch}, gradient accumulation steps: {args.acc_steps}')


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
  HF_TOKEN='XXX'
  DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  THRESHOLD = args.threshold
  ROOT = os.getcwd()
  DATASET_TRAIN = ROOT + os.sep +'data/new_data/atc/'  + args.train_path
  DATASET_TEST = ROOT + os.sep +'data/new_data/atc/roabsa_test.csv'
  DATASET_VAL = ROOT + os.sep +'data/new_data/atc/roabsa_eval.csv'
  tag = args.model.split('/')[-1]
  version = str(args.version)
  MODEL_SAVE_PATH = ROOT + os.sep + f'models/new_models/atc_{tag}_encdec_v{version}.pt' 
  MODEL_PRETRAINED_ATE = args.model 
  PRE_TRAINED_TOKENIZER_NAME = args.model
  WANDB_INIT_NAME = args.model 
  

  MAX_SOURCE_LEN = 512
  MAX_TARGET_LEN = 30
  
  BATCH_SIZE = args.batch
  BATCH_SIZE_TEST = 2
  
  EPOCHS = args.epochs
  LR = [3e-5, 1e-4, 2e-4, 3e-4]
  LR_IDX = 0
  EPS = 1e-5
  
  CATS = [
    "product","shop diversity","staff competency","shop organization","service",
    "quality","price","environment","staff availability","misc","delivery",
    "promotions","tech support","return warranty","security"
        ]
  
  COUNTS = {
    'product': 180, 'shop diversity': 135, 'staff competency': 132, 'shop organization': 99,
    'service': 95, 'quality': 82, 'price': 53, 'environment': 38, 'staff availability': 31,
    'misc': 27, 'delivery': 25, 'promotions': 15, 'tech support': 14, 'return warranty': 13, 'security': 2
    }
  
  cat2id = {c:i for i,c in enumerate(CATS)}
  id2cat = {i:c for c,i in cat2id.items()}

  gradient_accumulation_steps = args.acc_steps
  label_smoothing = 0.1 
  
  USE_LABEL_SMOOTHING = False
  if gradient_accumulation_steps > 0:
    USE_GRADIENT_ACC = True
  else:
    USE_GRADIENT_ACC = False

  USE_CONSTANT_SCHEDULER = True


login(token=config.HF_TOKEN)

wandb.init(project="final_new_roabsa_atc", name=config.WANDB_INIT_NAME,
            config={
                    "learning rate": config.LR[config.LR_IDX],
                    "optimizer": "adamw",
                    "method": "encoder_decoder",
                    "use label smoothing": config.USE_LABEL_SMOOTHING,
                    "scheduler": args.scheduler,
                    "gradient acc": config.USE_GRADIENT_ACC,
                    "batch_size": config.BATCH_SIZE,
                    "MAX_SOURCE_LEN": config.MAX_SOURCE_LEN,
                    "MAX_TARGET_LEN": config.MAX_TARGET_LEN,
                    "epochs": config.EPOCHS,
                    "dataset train": f"{config.DATASET_TRAIN}",
                    "dataset test": f"{config.DATASET_TEST}",
                    "dataset val": f"{config.DATASET_VAL}",
                    "pretrained model":config.MODEL_PRETRAINED_ATE,
                    "pretrained tokenizer":config.PRE_TRAINED_TOKENIZER_NAME,
                    "model save path": config.MODEL_SAVE_PATH,
                })

set_seed(config.SEED)
df_train = pd.read_csv(config.DATASET_TRAIN)
df_test = pd.read_csv(config.DATASET_TEST)
df_val = pd.read_csv(config.DATASET_VAL)

# df_train = df_train[:50]
# df_test = df_test[:20]
# df_val = df_val[:10]


def select_train_instances(df_train):
    if args.aug=='c-mlm':
        df_train = df_train[df_train['data_origin'].isin(['manual','c-mlm'])]

    if args.aug=='rc':
        df_train = df_train[df_train['data_origin'].isin(['manual','random-concatenation'])]

    if args.aug=='bt':
        df_train = df_train[df_train['data_origin'].isin(['manual','bt_1chain_RoFrRo','bt_2chain_RoFrChzRo','bt_2chain_RoEnChzRo'])]
    
    if args.aug=='bt+rc+c-mlm':
            df_train = df_train[df_train['data_origin'].isin(['manual','bt_1chain_RoFrRo','bt_2chain_RoFrChzRo','bt_2chain_RoEnChzRo','c-mlm','random-concatenation'])]
            
    if args.aug=='rephrasing':
        df_train = df_train[df_train['data_origin'].isin(['manual','rephrasing'])]

    if args.aug=='quad2text':
        df_train = df_train[df_train['data_origin'].isin(['manual','quad2text_trained'])]
    
    if args.aug=='rephrasing+quad2text':
        df_train = df_train[df_train['data_origin'].isin(['manual','quad2text_trained','rephrasing'])]
    if args.aug=='bt+quad2text':
        df_train = df_train[df_train['data_origin'].isin(['manual','quad2text_trained','bt_1chain_RoFrRo','bt_2chain_RoFrChzRo','bt_2chain_RoEnChzRo'])]
    if args.aug=='rephrasing+quad2text+bt':
        df_train = df_train[df_train['data_origin'].isin(['manual','quad2text_trained','rephrasing','bt_1chain_RoFrRo','bt_2chain_RoFrChzRo','bt_2chain_RoEnChzRo'])]
            
    return df_train


def remove_prompt(text):
    prompt = "Extract pairs of aspect categories with their corresponding opinions from the following Romanian review:"
    
    if prompt in text:
        text = text.replace(prompt,'')
    
    eos_token_t5 = '</s>'
    bos_token_t5 = '<s>'
    
    if eos_token_t5 in text:
        text = text.replace(eos_token_t5,'')
    
    if bos_token_t5 in text:
        text = text.replace(bos_token_t5,'')
        
    return text.strip()

df_train['text_cleaned'] = df_train['text_cleaned'].apply(remove_prompt)
df_val['text_cleaned'] = df_val['text_cleaned'].apply(remove_prompt)
df_test['text_cleaned'] = df_test['text_cleaned'].apply(remove_prompt)

if args.aug and args.aug!='all':
    df_train = select_train_instances(df_train)


def format_text_for_df(df, text_col_name):
    df[text_col_name] = df[text_col_name].apply(lambda x: x.lower())
    return df

def parse_cats(s):
    if not isinstance(s, str): return []
    return [x.strip() for x in s.split(";") if x.strip()]

def to_multihot(cats):
    y = [0]*len(config.CATS)
    for c in cats:
        if c in config.cat2id:
            y[config.cat2id[c]] = 1
    return y

def compute_pos_weight_from_df(df, label_col="labels", cap=20.0):
    # df[label_col] is a list-of-ints multihot of length L
    Y = np.vstack(df[label_col].tolist())  
    N, L = Y.shape
    P = Y.sum(axis=0)                     
    P = np.maximum(P, 1.0)
    pos_w = (N - P) / P
    pos_w = np.clip(pos_w, 1.0, cap)
    return torch.tensor(pos_w, dtype=torch.float)


df_train["labels_list"] = df_train["all_categories"].apply(parse_cats)
df_val["labels_list"] = df_val["all_categories"].apply(parse_cats)
df_test["labels_list"] = df_test["all_categories"].apply(parse_cats)


df_train["labels"] = df_train["labels_list"].apply(to_multihot)
df_val["labels"] = df_val["labels_list"].apply(to_multihot)
df_test["labels"] = df_test["labels_list"].apply(to_multihot)

pos_weight = compute_pos_weight_from_df(df_train, label_col="labels")

df_train = format_text_for_df(df_train, 'text_cleaned')
df_test = format_text_for_df(df_test, 'text_cleaned')
df_val = format_text_for_df(df_val, 'text_cleaned')

def multihot_to_labels(row, cats):
    idx = np.where(row == 1)[0]
    return [cats[j] for j in idx]

def labels_to_string(labels):
    return "; ".join(labels)

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

def load_model_and_tokenizer( model_path: str,
                            token: str,
                            ) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    model_config =  AutoConfig.from_pretrained(
        model_path,
        num_labels = len(config.CATS),
        problem_type="multi_label_classification"
    )      
    model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                token=token,
                config = model_config
                )
   
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              use_fast=False,
                                              token=token)

    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(
                                            model_path=config.MODEL_PRETRAINED_ATE,
                                            token=config.HF_TOKEN
                                        )

class WeightedBCELossTrainer(Trainer):
    def __init__(self, pos_weight=None, **kwargs):
        super().__init__(**kwargs)
        print('Initializing loss function...')
        self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(config.DEVICE))
        print('Trainer initialized')

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)                  
        logits = outputs.logits
        loss = self.loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


def sigmoid(x): return 1 / (1 + np.exp(-x))


class ABSADataset(Dataset):
    def __init__(self, df, tokenizer):
        self.inputs = df["text_cleaned"].astype(str).tolist()
        self.targets = df["labels"].tolist()
        self.tokenizer = tokenizer
        self.max_src_length = config.MAX_SOURCE_LEN
        self.max_tgt_length = config.MAX_TARGET_LEN
        

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_tensor = torch.tensor(self.targets[idx], dtype=torch.float)
        
        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_src_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens = True
        )

        item = {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "labels": target_tensor
        }

        return item
 

train_dataset = ABSADataset(df_train, tokenizer)
val_dataset = ABSADataset(df_val, tokenizer)
collator = DataCollatorWithPadding(tokenizer=tokenizer,
                                  return_tensors = "pt",
                                ) 

def evaluate_multilabel(
    df_test,
    model,
    tokenizer,
    cats,                         # list of label names in model order
    thresholds=None,              # np.array shape [num_labels]; if None -> 0.5
    batch_size=32,
    max_length=256,
    text_col="text_cleaned",
    label_col="labels",
    log_to_wandb=True
):
    
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    f1_instance_level, recalls, precisions =[], [], []
    all_preds, all_labels = [], []

    device = config.DEVICE
    num_labels = len(cats)
    if thresholds is None:
        thresholds = np.full(num_labels, config.THRESHOLD, dtype=np.float32)
        
    test_ds = ABSADataset(df_test, tokenizer=tokenizer)
    loader = DataLoader(test_ds, batch_size=batch_size, 
                        shuffle=False, collate_fn=collator,
                        )

    model.eval()
    
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()  
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = sigmoid(logits.detach().cpu().numpy())
            preds = (probs >= thresholds).astype(int)
           
        pred_strs = [labels_to_string(multihot_to_labels(pred, cats)) for pred in preds]
        refs_strs = [labels_to_string(multihot_to_labels(label, cats)) for label in labels]
        print('Current str predictions:', pred_strs)
        print('Current str references:', refs_strs)
        
        for pred, ref in zip(pred_strs, refs_strs):
            current_recall = recall(pred=pred, target=ref)
            current_precision = precision(pred=pred, target=ref)
            recalls.append(current_recall)
            precisions.append(current_precision)
            f1_instance_level.append(label_f1(precision=current_precision, recall=current_recall))
            print('Current recall:', current_recall)
            print('Current precision:', current_precision)
            
        print()
        all_preds.extend(pred_strs)
        all_labels.extend(refs_strs)
        

    result_rouge = rouge_metric.compute(predictions=all_preds, references=all_labels)
    result_bleu = bleu_metric.compute(predictions=all_preds, references=[[ref] for ref in all_labels])
    result_f1 = f1_score(all_labels, all_preds, average='weighted')
    
   
    result_recall = np.mean(recalls)
    result_precision = np.mean(precisions)
    result_f1_instance_level = np.mean(f1_instance_level)

        
    print("\nROUGE:", result_rouge)
    print("BLEU:", result_bleu['bleu'])
    print("BLEU precisions:", result_bleu['precisions'])
    print("F1:", result_f1)
    print("Recall:", result_recall)
    print("Precision:", result_precision)
    print("Result F1 instance level:", result_f1_instance_level)

    wandb.log({
        'Test Rouge R1': result_rouge['rouge1'],
        'Test Rouge R2': result_rouge['rouge2'],
        'Test Rouge L': result_rouge['rougeL'],
        'Test Bleu': result_bleu['bleu'],
        'Test Bleu precisions': np.mean(result_bleu['precisions']), 
        'Test F1': result_f1,
        'Test Recall': result_recall,
        'Test Precision': result_precision,
        'Test F1 instance level': result_f1_instance_level
    })
    
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,  
    early_stopping_threshold=0.01  
)


training_args = TrainingArguments(
        output_dir=f"logs_{config.WANDB_INIT_NAME}",
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE_TEST,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.LR[config.LR_IDX],
        logging_steps=10,
        num_train_epochs=config.EPOCHS,
        optim="adamw_8bit",
        report_to="wandb",
        run_name=config.WANDB_INIT_NAME,
        lr_scheduler_type=args.scheduler, # change to cosine
        weight_decay=0.01,
        max_grad_norm=1.0, 
        warmup_ratio=0.03, 
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        evaluation_strategy="epoch",
        use_cpu=False,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        disable_tqdm=False,
        group_by_length=False,
        dataloader_drop_last=False,
        dataloader_num_workers=8,
        remove_unused_columns=False,  
    )

try:
    print('Initialising trainer..')
    trainer = WeightedBCELossTrainer(train_dataset=train_dataset,
                                    eval_dataset=val_dataset,
                                    model=model,
                                    tokenizer=tokenizer,
                                    args=training_args,
                                    data_collator=collator,
                                    callbacks=[early_stopping],
                                    pos_weight=pos_weight 
                                )
    
    print('Start training...')
    trainer.train()
    print('Training finished')
    trainer.save_model(config.MODEL_SAVE_PATH)
    print(f'Model saved to: {config.MODEL_SAVE_PATH}')
     
    del model
    del train_dataset
    del val_dataset
    gc.collect()
    torch.cuda.empty_cache()  
    
    print('Testing the model...')
    evaluate_multilabel(df_test=df_test,
                        model = trainer.model,
                        tokenizer=tokenizer,
                        cats=config.CATS,
                        thresholds = None,
                        batch_size=config.BATCH_SIZE_TEST,
                        max_length = config.MAX_TARGET_LEN)

    print('Evaluation finished')
    wandb.finish()
except Exception as e:
  print(e)