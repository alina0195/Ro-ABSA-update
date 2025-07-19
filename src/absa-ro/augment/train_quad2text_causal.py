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
import argparse
from peft import LoraConfig
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from huggingface_hub import login
login(token="xxx")

nltk.download('punkt')
torch.set_warn_always(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description="Script configuration")
parser.add_argument("--model", type=str, default='', help="Pretrained model name")
parser.add_argument("--train_path", type=str, default='train_absaPairs_aug_v3_entailed.csv', help="Train csv base name")
parser.add_argument("--batch", type=int, default=4, help="Batch size")
parser.add_argument("--acc_steps", type=int, default=2, help="Gradient accumulation steps")
parser.add_argument("--use_lora", type=str, default='yes', help="Train lora. Options: yes/no")
parser.add_argument("--epochs", type=int, default=10, help="Number of Epochs")
parser.add_argument("--version", type=int, default=3, help="Model version for saving")
parser.add_argument("--description", type=str, default='Train to Generate text given the target. Training is the original set without rare categories. Test set original is not used. Test set is the rare set from train.', help="describe the setup")
args = parser.parse_args()
print(f'Task Running with batch size: {args.batch}, gradient accumulation steps: {args.acc_steps}, train df: {args.train_path}')

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
  DATASET_TEST = ROOT + os.sep +'data/new_data/absa/df_with_only_rare_lbls.csv'
  DATASET_TRAIN_ORIG = ROOT + os.sep + 'data/new_data/absa/' + args.train_path
  DATASET_TEST_ORIG = ROOT + os.sep +'data/new_data/absa/test_absaPairs.csv'
  DATASET_VAL_ORIG = ROOT + os.sep +'data/new_data/absa/eval_absaPairs.csv'
  
  MODEL_PRETRAINED_ATE = args.model  
  tag = MODEL_PRETRAINED_ATE.split('/')[-1]
  MODEL_SAVE_PATH = ROOT + os.sep + f'models/new_models/absa_quad2text_v{args.version}.pt' 
  PATH_SAVE_RESULTS = ROOT + os.sep + f'data/new_data/absa/predictions_trainedQuad2Text_on_rarelbls_{tag}.csv' 
  
  PRE_TRAINED_TOKENIZER_NAME =args.model
  WANDB_INIT_NAME = f"{args.model}" 
  
  MAX_SOURCE_LEN = 40
  MAX_TARGET_LEN = 256
  
  BATCH_SIZE = args.batch
  BATCH_SIZE_TEST = 4
  
  EPOCHS = args.epochs
  LR = [3e-5, 1e-4, 2e-4, 3e-4]
  LR_IDX = 0
  EPS = 1e-5
  
  gradient_accumulation_steps = args.acc_steps
  label_smoothing = 0.1 
  
  USE_LABEL_SMOOTHING = False
  
  if args.use_lora=='yes':
    USE_LORA = True
  else:
    USE_LORA = False
      
  USE_CONSTANT_SCHEDULER = False

use_cpu = False
resized_embeddings = False

gen_config = GenerationConfig(
    max_new_tokens = config.MAX_TARGET_LEN,
    return_dict_in_generate=True,
    output_scores=True,
    length_penalty=0,
    do_sample=True,
    temperature=0.9,
    top_p=0.8,
    repetition_penalty=1.2,
    num_return_sequences=1,
    no_repeat_ngram_size=10
    
)

wandb.init(project="quad2text", name=config.WANDB_INIT_NAME,
            config={
                    "Description": args.description,
                    "learning rate": config.LR[config.LR_IDX],
                    "optimizer": "adamw",
                    "use label smoothing": config.USE_LABEL_SMOOTHING,
                    "constant scheduler": config.USE_CONSTANT_SCHEDULER,
                    "lora": config.USE_LORA,
                    "gradient acc": config.gradient_accumulation_steps,
                    "batch_size": config.BATCH_SIZE,
                    "MAX_SOURCE_LEN": config.MAX_SOURCE_LEN,
                    "MAX_TARGET_LEN": config.MAX_TARGET_LEN,
                    "epochs": config.EPOCHS,
                    "dataset train ORIG": f"{config.DATASET_TRAIN_ORIG}",
                    "dataset test ORIG": f"{config.DATASET_TEST_ORIG}",
                    "dataset test": f"{config.DATASET_TEST}",
                    "dataset val ORIG": f"{config.DATASET_VAL_ORIG}",
                    "pretrained model":config.MODEL_PRETRAINED_ATE,
                    "pretrained tokenizer":config.PRE_TRAINED_TOKENIZER_NAME,
                    "model save path": config.MODEL_SAVE_PATH,
                    "Generate do_sample": gen_config.do_sample,
                    "Generate temperature": gen_config.temperature,
                    "Generate beams": gen_config.num_beams,
                    "Generate length penalty": gen_config.length_penalty,
                    "Generate repetition_penalty": gen_config.repetition_penalty,
                    "Generate no_repeat_ngram_size": gen_config.no_repeat_ngram_size,
                })

set_seed(config.SEED)
df_train_orig = pd.read_csv(config.DATASET_TRAIN_ORIG)
df_val = pd.read_csv(config.DATASET_VAL_ORIG)
df_test = pd.read_csv(config.DATASET_TEST)

rare_mask = df_train_orig['absa_input'].isin(df_test['absa_input'])
df_train = df_train_orig[~rare_mask].reset_index(drop=True)

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

df_train.rename(columns={'absa_input':'target','absa_target':'input'}, inplace=True)
df_val.rename(columns={'absa_input':'target','absa_target':'input'}, inplace=True)
df_test.rename(columns={'absa_input':'target','absa_target':'input'}, inplace=True)


def load_model_and_tokenizer(
                                model_path: str,
                                load_in_8bits: bool,
                                token: str,
                                use_cpu: bool = False,
                            ) -> tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
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
                load_in_4bit=True, 
                bnb_4bit_quant_type="nf4",  
                bnb_4bit_use_double_quant=True,  # Using double quantization as mentioned in QLoRA paper
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            print("Loading model in 4-bit mode")
    else:
        print("Using CPU")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=token,
        low_cpu_mem_usage=True if not use_cpu else False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              use_fast=False,
                                              padding_side="left", 
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
    sys_prompt =  "You are a customer passioned about writing online reviews for feedback. Starting from a list with the aspect categories and their sentiment associated, write a review which reflect all of them and nothing more."
          
    messages =  [  
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": example["input"]}, 
            {"role": "assistant", "content": example["target"]},  
        ] 
    
    formatted_text = tokenizer.apply_chat_template(messages, 
                                                    tokenize=False, 
                                                    add_special_tokens=False,
                                                    add_generation_prompt=False)
    
    return formatted_text

def format_as_chat_without_output(absa_input, add_system_msg=True):
    sys_prompt =  "You are a customer passioned about writing online reviews for feedback. Starting from a list with the aspect categories and their sentiment associated, write a review which reflect all of them and nothing more."
    
    system_message = {
        "role": "system",
        "content": sys_prompt
    }
    messages =  [system_message, {"role": "user", "content": absa_input}] # good for llama
        
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
            'target': item['target'],
            'input': item['input'],
            'input_ids' : item['input_ids'],
            'attention_mask' : item['attention_mask'],
        }
        
def tokenize_test_example(example):
    tokenized = tokenizer(example, 
                          padding=False,  # We'll pad later in collate_fn
                          return_tensors="pt",
                          truncation=True)
    return tokenized['input_ids'], tokenized['attention_mask']

def collate_test_fn(batch):
    input_ids = [ex["input_ids"] for ex in batch]
    attention_masks = [ex["attention_mask"] for ex in batch]

    padded = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_masks},
        padding=True,          # pad to max length in this batch
        return_tensors="pt"
    )
    return {
        "target": [ex["target"] for ex in batch],
        "input": [ex["input"] for ex in batch],
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"]
    }


def evaluate_model(df_test, model, tokenizer):
  bleu_metric = evaluate.load("bleu")
  rouge_metric = evaluate.load("rouge")
  
  df_test['text_formatted'] = df_test['input'].apply(format_as_chat_without_output)
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
  all_inputs =[]
  
  for batch in test_dataloader: 
    input_ids = batch['input_ids'].to(model.device)
    attention_mask = batch['attention_mask'].to(model.device)
    references = batch['target']
    inputs = batch['input']
    
    with torch.no_grad():
      outputs = model.generate(input_ids=input_ids,
                               attention_mask= attention_mask,
                                max_new_tokens = config.MAX_TARGET_LEN,
                                return_dict_in_generate=True,
                                output_scores=True,
                                temperature=0.9,
                                top_p=0.8,
                                length_penalty=0,
                                do_sample=True,
                                repetition_penalty=1.2,
                                num_return_sequences=1,
                                no_repeat_ngram_size=10,
                                pad_token_id = tokenizer.pad_token_id,
                                early_stopping=False  
                               ) # GenerateBeamDecoderOnlyOutput
      
    predictions = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

    cleaned_outputs = []
    for text in predictions:
        chunks = text.split("assistant")
        if len(chunks)>1:
            text = chunks[-1].strip()
        
        cleaned_outputs.append(text)
    
    preds.extend(cleaned_outputs)
    refs.extend(references)
    all_inputs.extend(inputs)
    
    print("\nPrediction:", cleaned_outputs)
    print("Reference:", references)
    print("Input:", inputs)
    
  result_rouge = rouge_metric.compute(predictions=preds, 
                                      references=refs)
  result_bleu = bleu_metric.compute(predictions=preds, 
                                    references=[[ref] for ref in refs])
  
  with open("preds.txt", "w", encoding="utf-8") as f:
    for item in preds:
        f.write(f"{item}\n")      
  with open("inputs.txt", "w", encoding="utf-8") as f:
    for item in all_inputs:
        f.write(f"{item}\n")    
  with open("refs.txt", "w", encoding="utf-8") as f:
    for item in refs:
        f.write(f"{item}\n")   
  try:       
    predicted_pairs =pd.DataFrame({'absa_input':preds,'absa_target':all_inputs, 'text_reference':refs})
    predicted_pairs.to_csv(config.PATH_SAVE_RESULTS, index=False, encoding="utf-8")
    print(f"Saved {len(predicted_pairs):,} rows in {config.PATH_SAVE_RESULTS}")
  except:
      print('Could not save results as csv file')
  print("\nROUGE:", result_rouge)
  print("BLEU:", result_bleu['bleu'])
  print("BLEU precisions:", result_bleu['precisions'])
  
  wandb.log({
      'Test Rouge R1':result_rouge['rouge1'],
      'Test Rouge R2':result_rouge['rouge2'],
      'Test Rouge L':result_rouge['rougeL'],
      'Test Bleu':result_bleu['bleu'],
      'Test Bleu precisions':np.mean(result_bleu['precisions']), 
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
        lr_scheduler_type="constant", # change to cosine
        weight_decay=0.01,
        max_grad_norm=0.1, #based on QLoRa paper
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