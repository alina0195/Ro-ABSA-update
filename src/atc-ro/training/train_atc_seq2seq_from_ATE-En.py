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
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase, GenerationConfig
from transformers import (BitsAndBytesConfig, 
                          EarlyStoppingCallback,
                          AutoTokenizer)
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import (LoraConfig, 
                  TaskType)           
from huggingface_hub import login

import argparse
login(token="xxx")
nltk.download('punkt')
torch.set_warn_always(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


parser = argparse.ArgumentParser(description="Script configuration")
parser.add_argument("--batch", type=int, default=4, help="Batch size")
parser.add_argument("--acc_steps", type=int, default=2, help="Gradient accumulation steps")
parser.add_argument("--use_lora", type=bool, default=False, help="Use Lora or Not")
parser.add_argument("--epochs", type=int, default=10, help="Number of Epochs")
parser.add_argument("--version", type=int, default=1, help="Version for model's name saved")

args = parser.parse_args()
print(f'Task Running for {args.epochs} epochs batch size: {args.batch}, gradient accumulation steps: {args.acc_steps}, using lora: {args.use_lora}')


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
  DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  
  ROOT = os.getcwd()
  DATASET_TRAIN = ROOT + os.sep +'data/new_data/atc/roabsa_train.csv'
  DATASET_TEST = ROOT + os.sep +'data/new_data/atc/roabsa_test.csv'
  DATASET_VAL = ROOT + os.sep +'data/new_data/atc/roabsa_eval.csv'
  version = str(args.version)
  MODEL_SAVE_PATH = ROOT + os.sep + f'models/new_models/atc_seq2seq_v{version}_fromEn.pt' 
  MODEL_PRETRAINED_ATE = ROOT + os.sep + '/models/teacher_model' 
  PRE_TRAINED_TOKENIZER_NAME = 'google/flan-t5-large'
  WANDB_INIT_NAME = 'english-transfer' 
  MAX_SOURCE_LEN = 512
  MAX_TARGET_LEN = 30
  
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
     
  if args.use_lora=='yes': 
    USE_LORA = True
  else:
    USE_LORA = False
      
  USE_CONSTANT_SCHEDULER = True

use_cpu = False
resized_embeddings = False
gen_config = GenerationConfig(
    max_new_tokens = config.MAX_TARGET_LEN,
    return_dict_in_generate=True,
    output_scores=True,
    num_beams=20,
    length_penalty=0,
    do_sample=True,
    temperature=0.3,
    repetition_penalty=1.2,
    num_return_sequences=1,
)

wandb.init(project="new_roabsa_atc", name=config.WANDB_INIT_NAME,
            config={
                    "Description": "Train ATC model with output pairs: <ATC1; ...;ATCn>",
                    "learning rate": config.LR[config.LR_IDX],
                    "optimizer": "adam",
                    "use label smoothing": config.USE_LABEL_SMOOTHING,
                    "constant scheduler": config.USE_CONSTANT_SCHEDULER,
                    "lora": config.USE_LORA,
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
                    "Accelerate used": "False",
                    "Generate do_sample": gen_config.do_sample,
                    "Generate temperature": gen_config.temperature,
                    "Generate beams": gen_config.num_beams,
                    "Generate length penalty": gen_config.length_penalty,
                    "Generate repetition_penalty": gen_config.repetition_penalty,
                })

set_seed(config.SEED)
df_train = pd.read_csv(config.DATASET_TRAIN)
df_test = pd.read_csv(config.DATASET_TEST)
df_val = pd.read_csv(config.DATASET_VAL)


def format_text_for_df(df, text_col_name):
    df[text_col_name] = df[text_col_name].apply(lambda x: x.lower())
    df[text_col_name] = df[text_col_name] + ' </s>'
    return df


df_train = format_text_for_df(df_train, 'text_cleaned')
df_test = format_text_for_df(df_test, 'text_cleaned')
df_val = format_text_for_df(df_val, 'text_cleaned')


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

def load_model_and_tokenizer(
                                model_path: str,
                                load_in_8bits: bool,
                                token: str,
                                use_cpu: bool = False,
                            ) -> tuple[AutoModelForSeq2SeqLM, PreTrainedTokenizerBase]:
                                
    bnb_config = None
    if not use_cpu:
        if load_in_8bits:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,  # load model in 8-bit precision
                low_cpu_mem_usage=True,
            )
            print("Loading model in 8-bit mode")
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,  # load model in 4-bit precision
                bnb_4bit_quant_type="nf4",  # pre-trained model should be quantized in 4-bit NF format
                bnb_4bit_use_double_quant=True,  # Using double quantization as mentioned in QLoRA paper
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            print("Loading model in 4-bit mode")
    else:
        print("Using CPU")
    
    if not config.USE_LORA:
        model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                # quantization_config=bnb_config,
                token=token,
                low_cpu_mem_usage=True if not use_cpu else False,
                )
        lora_config=None
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                token=token,
                low_cpu_mem_usage=True if not use_cpu else False,
                )

        lora_config = LoraConfig(
                        r=64,
                        lora_alpha=16,
                        target_modules = ["q", "v"],  
                        lora_dropout=0.1,
                        bias="none",
                        modules_to_save= ["lm_head", "embed_tokens"],  # keep these in fp16
                        task_type=TaskType.SEQ_2_SEQ_LM,
                        use_rslora=True,
                    )

    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_TOKENIZER_NAME,
                                              use_fast=False,
                                              token=token)

    return model, tokenizer, lora_config

model, tokenizer, lora_config = load_model_and_tokenizer(
                                            model_path=config.MODEL_PRETRAINED_ATE,
                                            load_in_8bits= True,
                                            token='xxx',
                                            use_cpu= False)

print("PAD Token:", tokenizer.pad_token)
print("PAD Token ID:", tokenizer.pad_token_id)
print("EOS Token:", tokenizer.eos_token)
print("EOS Token ID:", tokenizer.eos_token_id)
print("Padding Side:", tokenizer.padding_side)

def test_tokenizer():
    vocab_size = tokenizer.vocab_size
    unused_token_id = vocab_size  # Typically, the next available ID is safe

    print("Vocab Size:", vocab_size)
    print("Unused Token ID:", unused_token_id)
    
    test_texts = [
    "The laptop's battery lasts very long, but the screen is dim. </s>",
    "I love the fast delivery! </s>",
    "The food was terrible, but the service was excellent."
    ]

    # Tokenize with padding enabled
    tokenized = tokenizer(
        test_texts, 
        padding=True,  # Pad to longest sequence
        truncation=True, 
        return_tensors="pt",
        add_special_tokens=True
    )

    print("Tokenized Input IDs (TRUE):", tokenized["input_ids"])
    print("Decoded Texts (TRUE):", tokenizer.batch_decode(tokenized["input_ids"]))
    print("Attention Mask (TRUE):", tokenized["attention_mask"])

    tokenized = tokenizer(
        test_texts, 
        padding=True,  # Pad to longest sequence
        truncation=True, 
        return_tensors="pt",
        add_special_tokens=False
    )

    print("Tokenized Input IDs (FALSE):", tokenized["input_ids"])
    print("Decoded Texts (FALSE):", tokenizer.batch_decode(tokenized["input_ids"]))
    print("Attention Mask (FALSE):", tokenized["attention_mask"])
    
    
test_tokenizer()


def tokenize_input(example, padding):
    tokenizer.target_tokenizer = False
    tokenized = tokenizer(example, 
                          padding=padding,  # padded later in collate_test_fn
                          return_tensors="pt",
                          truncation=True)
    return tokenized['input_ids'], tokenized['attention_mask']

def tokenize_target(example):
    tokenizer.target_tokenizer = True
    tokenized = tokenizer(example, 
                          padding=False,  
                          return_tensors="pt",
                          truncation=True)
    return tokenized['input_ids']


collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                  model=model,
                                  return_tensors = "pt",
                                ) 
        
class ABSADataset(Dataset):
    def __init__(self, df, tokenizer):
        self.inputs = df["text_cleaned"].tolist()
        self.targets = df["all_categories_old"].tolist()
        self.tokenizer = tokenizer
        self.max_src_length = config.MAX_SOURCE_LEN
        self.max_tgt_length = config.MAX_TARGET_LEN
        

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]

        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_src_length,
            # padding="max_length",
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens = False
        )

        
        labels = tokenizer(text_target=target_text, 
                           return_tensors="pt",
                           max_length=self.max_tgt_length,
                           truncation=True, 
                           add_special_tokens = True,
                           padding=False)
        item = {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "labels": labels["input_ids"].squeeze(0),
        }

        return item
 
        

train_dataset = ABSADataset(df_train, tokenizer)
val_dataset = ABSADataset(df_val, tokenizer)


def collate_test_fn(batch):
    input_ids = [ex["input_ids"] for ex in batch]
    attention_masks = [ex["attention_mask"] for ex in batch]
    refs = [ex["labels"] for ex in batch]

    padded = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_masks},
        padding=True,          # pad to max length in this batch
        return_tensors="pt"
    )
    return {
        "labels": refs,
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"]
    }


def safe_batch_decode(tokenizer, tensor):
    """
    Replace every ID that is < 0  or >= vocab_size by pad_token_id,
    then decode with skip_special_tokens=True.
    """
    if isinstance(tensor, torch.Tensor):
        ids = tensor.clone()                    # do NOT modify the original tensor
    else:                                       # list / nested-list
        ids = torch.tensor(tensor)

    bad = (ids < 0) | (ids >= tokenizer.vocab_size)
    ids[bad] = tokenizer.pad_token_id
    return tokenizer.batch_decode(ids, skip_special_tokens=True)


def evaluate_model(df_test, model, tokenizer):
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")

    test_dataset = ABSADataset(df_test, tokenizer)
    test_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE_TEST, 
        shuffle=False,
        collate_fn=test_collator
    )

    print('Evaluating model...')
    model.eval()

    preds, refs = [], []
    recalls, precisions, f1_instance_level = [], [], []
    
    for batch in test_dataloader: 
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels']

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.MAX_TARGET_LEN,
                return_dict_in_generate=True,
                output_scores=True,
                num_beams=20,
                temperature=0.3,
                length_penalty=0,
                do_sample=True,
                repetition_penalty=1.2,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                early_stopping=False
            )

        labels_for_decode = labels.clone()
        labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id

        decoded_preds = safe_batch_decode(tokenizer, outputs.sequences)
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = safe_batch_decode(tokenizer, labels)
        decoded_labels = [r.strip() for r in decoded_labels]

        preds.extend(decoded_preds)
        refs.extend(decoded_labels)

        print("Reference:", decoded_labels)
        print("Prediction:", decoded_preds)
        print()
        
        for pred, ref in zip(decoded_preds, decoded_labels):
            current_recall = recall(pred=pred, target=ref)
            current_precision = precision(pred=pred, target=ref)
            recalls.append(current_recall)
            precisions.append(current_precision)
            f1_instance_level.append(label_f1(precision=current_precision, recall=current_recall))

    result_rouge = rouge_metric.compute(predictions=preds, references=refs)
    result_bleu = bleu_metric.compute(predictions=preds, references=[[ref] for ref in refs])
    result_f1 = f1_score(refs, preds, average='weighted')
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


training_args = Seq2SeqTrainingArguments(
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
        lr_scheduler_type="constant", # change to cosine
        weight_decay=0.01,
        max_grad_norm=0.3, #based on QLoRa paper
        warmup_ratio=0.03, #based on QLoRa paper
        bf16=False,
        tf32=True if not use_cpu else False,
        save_strategy="epoch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        evaluation_strategy="epoch",
        use_cpu=False,
        remove_unused_columns=True,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        disable_tqdm=False,
        group_by_length=False,
        dataloader_drop_last=False,
        dataloader_num_workers=8,
        # predict_with_generate=True,
    )

try:
    print('Initialising trainer..')
    trainer = Seq2SeqTrainer(
                            train_dataset=train_dataset,
                            eval_dataset=val_dataset,
                            model=model,
                            tokenizer=tokenizer,
                            args=training_args,
                            data_collator=collator,
                            callbacks=[early_stopping]
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
    evaluate_model(df_test=df_test,
                   model=trainer.model,
                   tokenizer=tokenizer)
    print('Evaluation finished')
    wandb.finish()
except Exception as e:
  print(e)