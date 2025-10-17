import re
import wandb
import string
import evaluate
import random
import nltk
import os
import torch, gc
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from transformers import AutoModelForSequenceClassification, Trainer
from huggingface_hub import login
import requests
import argparse
login(token="xxx")
nltk.download('punkt')
torch.set_warn_always(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


parser = argparse.ArgumentParser(description="Script configuration")
parser.add_argument("--model", type=str, default='', help="Pretrained model name")
parser.add_argument("--batch", type=int, default=4, help="Batch size")
parser.add_argument("--acc_steps", type=int, default=2, help="Gradient accumulation steps")
parser.add_argument("--epochs", type=int, default=10, help="Number of Epochs")
parser.add_argument("--version", type=int, default=1, help="Version for model's name saved")
parser.add_argument("--train_path", type=str, default='train_absaPairs_aug_final.csv', help="Train csv base name")
parser.add_argument("--description", type=str, default='description', help="Describe the experiment")
parser.add_argument("--scheduler", type=str, default='constant', help="Type of scheduler")

args = parser.parse_args()
print(f'Task Running for model: {args.model}, batch size: {args.batch}, gradient accumulation steps: {args.acc_steps}, train path: {args.train_path}, scheduler: {args.scheduler}')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
 

# "microsoft/deberta-v3-base"

# ============================================================
# CONFIGURATION
# ============================================================
class config:
    SEED = 42
    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    ROOT = os.getcwd()
    DATASET_TRAIN = ROOT + os.sep + f'data/new_data/alsa/{args.train_path}'
    DATASET_TEST = ROOT + os.sep +'data/new_data/alsa/test_alsa.csv'
    DATASET_VAL = ROOT + os.sep +'data/new_data/alsa/val_alsa.csv'
    tag = args.model.split('/')[-1]
    version = str(args.version)
    MODEL_SAVE_PATH = ROOT + os.sep +f'models/new_models/alsa_{tag}_v{args.version}.pt' 
    MODEL_PRETRAINED_ATE = args.model 
    PRE_TRAINED_TOKENIZER_NAME = args.model
    WANDB_INIT_NAME = args.model 
    
    MAX_SOURCE_LEN = 512
    MAX_TARGET_LEN = 5
    
    BATCH_SIZE = args.batch
    BATCH_SIZE_TEST = 2
    
    EPOCHS = args.epochs
    LR = [3e-5, 1e-4, 2e-4, 3e-4]
    LR_IDX = 0
    EPS = 1e-5
    
    gradient_accumulation_steps = args.acc_steps
    label_smoothing = 0.1 
    
    USE_LABEL_SMOOTHING = False
    if gradient_accumulation_steps > 0:
        USE_GRADIENT_ACC = True
    else:
        USE_GRADIENT_ACC = False
        
    USE_CONSTANT_SCHEDULER = True
    LABELS = ["positive", "negative", "neutral"]



df_train = pd.read_csv(config.DATASET_TRAIN)
df_val = pd.read_csv(config.DATASET_VAL)
df_test = pd.read_csv(config.DATASET_TEST)

label2id = {label: i for i, label in enumerate(config.LABELS)}
id2label = {i: label for label, i in label2id.items()}

wandb.init(project="new_roabsa_alsa", name=config.WANDB_INIT_NAME,
            config={
                      "learning rate": config.LR[config.LR_IDX],
                      "optimizer": "adam",
                      "epochs": config.EPOCHS,
                      "batch": config.BATCH_SIZE,
                      "pretrained model":config.MODEL_PRETRAINED_ATE,
                      "pretrained tokenizer":config.PRE_TRAINED_TOKENIZER_NAME,
                      "model save path": config.MODEL_SAVE_PATH,
                      "scheduler": args.scheduler,
                      "MAX_SOURCE_LEN": config.MAX_SOURCE_LEN,
                      "MAX_TARGET_LEN": config.MAX_TARGET_LEN,
                      "dataset train": f"{config.DATASET_TRAIN}",
                      "dataset test": f"{config.DATASET_TEST}",
                      "dataset val": f"{config.DATASET_VAL}",
                      })
set_seed(config.SEED)


class ABSADataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["text_cleaned"].tolist()
        self.aspects = df["category"].tolist()
        self.labels = [label2id[p] for p in df["polarity"].tolist()]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        aspect = self.aspects[idx]
        label = self.labels[idx]

        input_text = f"{text}[SEP]{aspect}"

        encoded = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )

        item = {key: val.squeeze(0) for key, val in encoded.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item



tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PRETRAINED_ATE)

train_dataset = ABSADataset(df_train, tokenizer, config.MAX_SOURCE_LEN)
val_dataset = ABSADataset(df_val, tokenizer, config.MAX_SOURCE_LEN)
test_dataset = ABSADataset(df_test, tokenizer, config.MAX_SOURCE_LEN)


model = AutoModelForSequenceClassification.from_pretrained(
    config.MODEL_PRETRAINED_ATE,
    num_labels=len(config.LABELS),
    id2label=id2label,
    label2id=label2id
)

# 
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,  
    early_stopping_threshold=0.01  
)

training_args = TrainingArguments(
    output_dir=f"logs_{config.WANDB_INIT_NAME}",
    run_name=config.WANDB_INIT_NAME,
    evaluation_strategy="epoch",
    lr_scheduler_type=args.scheduler, 
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    per_device_train_batch_size=config.BATCH_SIZE,
    per_device_eval_batch_size=config.BATCH_SIZE,
    learning_rate=config.LR[config.LR_IDX],
    num_train_epochs=config.EPOCHS,
    logging_steps=50,
    save_total_limit=1,
    weight_decay=0.01,
    disable_tqdm=False,
    report_to="wandb",
    dataloader_num_workers=8,
    dataloader_drop_last=False,
    remove_unused_columns=True,
    use_cpu=False,
    optim="adamw_8bit",
)

print('Initialising trainer..')
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping]
    
)

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
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)
true = predictions.label_ids

print("\n--- TEST RESULTS ---")
print("Accuracy:", accuracy_score(true, preds))
print("Weighted F1:", f1_score(true, preds, average="weighted"))
print("\nClassification Report:")
print(classification_report(true, preds, target_names=config.LABELS))