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

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import (BitsAndBytesConfig, 
                          TrainingArguments,
                          )
from peft import (LoraConfig, 
                  TaskType, 
                  PeftModel)           

from transformers import EarlyStoppingCallback

from peft import LoraConfig
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from huggingface_hub import login
login(token="xxx")
import argparse

nltk.download('punkt')
torch.set_warn_always(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


parser = argparse.ArgumentParser(description="Script configuration")
parser.add_argument("--train_path", type=str, default='train_absaPairs_aug_final.csv', help="Train csv base name")
parser.add_argument("--batch", type=int, default=4, help="Batch size")
parser.add_argument("--acc_steps", type=int, default=2, help="Gradient accumulation steps")
parser.add_argument("--epochs", type=int, default=10, help="Number of Epochs")
parser.add_argument("--version", type=int, default=2, help="Model version for saving")
parser.add_argument("--aug", type=str, default='', help="Augmentation type to include. Values: all/bt/c-mlm/rc/rephrasing/quad2text_trained/rephrasing+quad2text")
parser.add_argument("--description", type=str, default='', help="Will be placed in wandb run description")
args = parser.parse_args()
print(f'Task Running batch size: {args.batch}, gradient accumulation steps: {args.acc_steps}, train path: {args.train_path}')


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
  DATASET_TRAIN = ROOT + os.sep + 'data/new_data/absa/' + args.train_path

  DATASET_TEST = ROOT + os.sep +'data/new_data/absa/test_absaPairs.csv'
  DATASET_VAL = ROOT + os.sep +'data/new_data/absa/eval_absaPairs.csv'
  
  MODEL_SAVE_PATH = ROOT + os.sep + f'models/new_models/final_absa_onego_rollama3.1-8b-instruct_{args.aug}_v{args.version}.pt' 
  MODEL_PRETRAINED_ATE ="OpenLLM-Ro/RoLlama3.1-8b-Instruct"
  PRE_TRAINED_TOKENIZER_NAME ="OpenLLM-Ro/RoLlama3.1-8b-Instruct"
  WANDB_INIT_NAME = f"{args.aug}_OpenLLM-Ro/RoLlama3.1-8b-Instruct"
  
  MAX_SOURCE_LEN = 512
  MAX_TARGET_LEN = 30
  
  BATCH_SIZE = args.batch
  BATCH_SIZE_TEST = 2
  
  EPOCHS = args.epochs
  LR = [5e-6, 3e-5, 1e-4, 2e-4, 3e-4]
  LR_IDX = 0
  EPS = 1e-5
  
  gradient_accumulation_steps = args.acc_steps
  label_smoothing = 0.1 
  
  USE_LABEL_SMOOTHING = False
  USE_GRADIENT_ACC = False
  USE_LORA = True
  USE_CONSTANT_SCHEDULER = False

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

wandb.init(project="final_new_roabsa_onego", name=config.WANDB_INIT_NAME,
            config={
                    "learning rate": config.LR[config.LR_IDX],
                    "optimizer": "adamw",
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

def select_train_instances(df_train):
    if args.aug=='c-mlm':
        df_train = df_train[df_train['data_origin'].isin(['manual','c-mlm'])]

    if args.aug=='rc':
        df_train = df_train[df_train['data_origin'].isin(['manual','random-concatenation'])]

    if args.aug=='bt':
        df_train = df_train[df_train['data_origin'].isin(['manual','bt_1chain_RoFrRo','bt_2chain_RoFrChzRo','bt_2chain_RoEnChzRo'])]
        
    if args.aug=='rephrasing':
        df_train = df_train[df_train['data_origin'].isin(['manual','rephrasing'])]

    if args.aug=='quad2text':
        df_train = df_train[df_train['data_origin'].isin(['manual','quad2text_trained'])]
    
    if args.aug=='rephrasing+quad2text':
        df_train = df_train[df_train['data_origin'].isin(['manual','quad2text_trained','rephrasing'])]
        
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

df_train['absa_input'] = df_train['absa_input'].apply(remove_prompt)
df_val['absa_input'] = df_val['absa_input'].apply(remove_prompt)
df_test['absa_input'] = df_test['absa_input'].apply(remove_prompt)

if args.aug and args.aug!='all':
    df_train = select_train_instances(df_train)
    
print('DF TRAIN:', len(df_train))
print('DF TEST:', len(df_test))
print('DF VAL:', len(df_val))

print('\n',df_train['data_origin'].value_counts(),'\n\n')

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
        return 0  
    return 2 * (precision * recall) / (precision + recall)

def load_model_and_tokenizer(
                                model_path: str,
                                load_in_8bits: bool,
                                token: str,
                                use_cpu: bool = False,
                            ) -> tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
                                
    bnb_config = None
    if not use_cpu:
        if load_in_8bits:  # this is our call by default 
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

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # quantization_config=bnb_config,
        token=token,
        low_cpu_mem_usage=True if not use_cpu else False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              use_fast=False,
                                              padding_side="left", #added
                                              token=token)

    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(
                                            model_path=config.MODEL_PRETRAINED_ATE,
                                            load_in_8bits= True,
                                            token='xxx',
                                            use_cpu= False)

tokenizer.pad_token_id = 128002  
print("PAD Token:", tokenizer.pad_token)
print("PAD Token ID:", tokenizer.pad_token_id)
print("EOS Token:", tokenizer.eos_token)
print("EOS Token ID:", tokenizer.eos_token_id)
print("Padding Side:", tokenizer.padding_side)
model.generation_config.pad_token_id = tokenizer.pad_token_id


def test_tokenizer():
    # Get the vocabulary size
    vocab_size = tokenizer.vocab_size

    # Find an unused token ID
    unused_token_id = vocab_size  # Typically, the next available ID is safe

    print("Vocab Size:", vocab_size)
    print("Unused Token ID:", unused_token_id)
    
    test_texts = [
    "The laptop's battery lasts very long, but the screen is dim.",
    "I love the fast delivery!",
    "The food was terrible, but the service was excellent."
    ]

    # Tokenize with padding enabled
    tokenized = tokenizer(
        test_texts, 
        padding=True,  # Pad to longest sequence
        truncation=True, 
        return_tensors="pt"
    )

    print("Tokenized Input IDs:", tokenized["input_ids"])
    print("Decoded Texts:", tokenizer.batch_decode(tokenized["input_ids"]))
    print("Attention Mask:", tokenized["attention_mask"])

test_tokenizer()

lora_config = LoraConfig(
                        r=64,
                        lora_alpha=16,
                        target_modules=['self_attn.q_proj','self_attn.v_proj'],  
                        lora_dropout=0.1,
                        bias="none",
                        modules_to_save=None if not resized_embeddings else ["lm_head", "embed_tokens"],
                        task_type=TaskType.CAUSAL_LM,
                        use_rslora=True,
                    )


collator = DataCollatorForCompletionOnlyLM(
                                    tokenizer=tokenizer,
                                    mlm=False,
                                    return_tensors="pt",
                                    response_template="<|start_header_id|>assistant<|end_header_id|>",
                                    ignore_index=-100
                                ) 

train_dataset = Dataset.from_pandas(df_train)
val_dataset = Dataset.from_pandas(df_val)


def format_as_chat_with_output(example):
    messages =  [  
            {"role": "system", "content": "You are an expert linguist assistant specialized in extracting pairs of categories with their sentiment polarity from  a review."},
            {"role": "user", "content": example["absa_input"]}, 
            {"role": "assistant", "content": example["absa_target"]},  
        ] 
    
    formatted_text = tokenizer.apply_chat_template(messages, 
                                                    tokenize=False, 
                                                    add_special_tokens=False,
                                                    add_generation_prompt=False)
    
    return formatted_text

def format_as_chat_without_output(absa_input, add_system_msg=True):
    if add_system_msg:
        system_message = {
            "role": "system",
            "content": "You are an expert linguist assistant specialized in extracting pairs of categories with their sentiment polarity from  a review."
        }
        messages =  [system_message, {"role": "user", "content": absa_input}] 
    else:
        messages =  [ {"role": "user", "content": absa_input}] 
        
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print('TEST Formatted text:', formatted_text)
    return formatted_text

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        return {
            'absa_target': item['absa_target'],
            'input_ids' : item['input_ids'],
            'attention_mask' : item['attention_mask'],
        }
        
def tokenize_test_example(example):
    tokenized = tokenizer(example, 
                          padding=False, 
                          return_tensors="pt",
                          truncation=True)
    return tokenized['input_ids'], tokenized['attention_mask']

def collate_test_fn(batch):
    input_ids = [ex["input_ids"] for ex in batch]
    attention_masks = [ex["attention_mask"] for ex in batch]
    refs = [ex["absa_target"] for ex in batch]

    padded = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_masks},
        padding=True,          # pad to max length in this batch
        return_tensors="pt"
    )
    return {
        "absa_target": refs,
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"]
    }


def evaluate_model(df_test, model, tokenizer):
  bleu_metric = evaluate.load("bleu")
  rouge_metric = evaluate.load("rouge")
  
  df_test['text_formatted'] = df_test['absa_input'].apply(format_as_chat_without_output)
  tokenized_outputs = tokenizer(df_test["text_formatted"].tolist(),
                                padding=True, 
                                truncation=True,
                                return_tensors="pt")
  df_test["input_ids"] = tokenized_outputs["input_ids"].tolist()
  df_test["attention_mask"] = tokenized_outputs["attention_mask"].tolist()
  
  test_dataloader = DataLoader(TestDataset(df_test), 
                             batch_size=config.BATCH_SIZE_TEST,
                             collate_fn=collate_test_fn)

  print('Evaluating model...')
  model.eval()

  preds = []
  refs = []
  
  recalls = []
  precisions = []
  f1_instance_level = []
  
  for batch in test_dataloader: 
    input_ids = batch['input_ids'].to(model.device)
    attention_mask = batch['attention_mask'].to(model.device)
    references = batch['absa_target']
    
    with torch.no_grad():
      outputs = model.generate(input_ids=input_ids,
                               attention_mask= attention_mask,
                                max_new_tokens = config.MAX_TARGET_LEN,
                                return_dict_in_generate=True,
                                output_scores=True,
                                num_beams=20,
                                temperature=0.3,
                                length_penalty=0,
                                do_sample=True,
                                repetition_penalty=1.2,
                                num_return_sequences=1,
                                pad_token_id = tokenizer.pad_token_id,
                                early_stopping=False  
                               ) 
      
    predictions = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

    cleaned_outputs = []
    for text in predictions:
        chunks = text.split("assistant")
        if len(chunks)>1:
            text = chunks[-1].strip()
        
        cleaned_outputs.append(text)
    
    preds.extend(cleaned_outputs)
    refs.extend(references)
    
    print("\nPrediction:", cleaned_outputs)
    print("Reference:", references)
    
    for pred, ref in zip(cleaned_outputs, references):
        current_recall = recall(pred=pred, target=ref)
        current_precision = precision(pred=pred, target=ref)
        recalls.append(current_recall)
        precisions.append(current_precision)
        f1_instance_level.append(label_f1(precision=current_precision, recall=current_recall))
        

  result_rouge = rouge_metric.compute(predictions=preds, 
                                      references=refs)
  result_bleu = bleu_metric.compute(predictions=preds, 
                                    references=[[ref] for ref in refs])
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
      'Test Rouge R1':result_rouge['rouge1'],
      'Test Rouge R2':result_rouge['rouge2'],
      'Test Rouge L':result_rouge['rougeL'],
      'Test Bleu':result_bleu['bleu'],
      'Test Bleu precisions':np.mean(result_bleu['precisions']), 
      'Test F1':result_f1,
      'Test Recall':result_recall,
      'Test Precision': result_precision,
      'Test F1 instance level': result_f1_instance_level
  })


early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,  
    early_stopping_threshold=0.01  
)


training_args = TrainingArguments(
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE_TEST,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.LR[config.LR_IDX],
        logging_steps=10,
        num_train_epochs=config.EPOCHS,
        optim="adamw_8bit",
        report_to="wandb",
        run_name=config.WANDB_INIT_NAME,
        lr_scheduler_type="constant", 
        weight_decay=0.01,
        max_grad_norm=1.0, #based on QLoRa paper
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
    )

try:
    print('Initialising trainer..')
    trainer = SFTTrainer(
                            train_dataset=train_dataset,
                            eval_dataset=val_dataset,
                            model=model,
                            peft_config=lora_config,
                            tokenizer=tokenizer,
                            args=training_args,
                            data_collator=collator,
                            formatting_func=format_as_chat_with_output,  
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