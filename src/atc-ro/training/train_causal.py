import datetime
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
                  )           

from transformers import EarlyStoppingCallback

from peft import LoraConfig
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from huggingface_hub import login
login(token="xxx")

nltk.download('punkt')
torch.set_warn_always(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse

parser = argparse.ArgumentParser(description="Script configuration")
parser.add_argument("--model", type=str, default='', help="Pretrained model name") # 
parser.add_argument("--batch", type=int, default=4, help="Batch size")
parser.add_argument("--acc_steps", type=int, default=2, help="Gradient accumulation steps")
parser.add_argument("--use_lora", type=str, default='no', help="Use Lora or Not")
parser.add_argument("--epochs", type=int, default=10, help="Number of Epochs")
parser.add_argument("--version", type=int, default=3, help="Version for model's name saved")
parser.add_argument("--description", type=str, default='', help="Will be placed in wandb run description")
parser.add_argument("--aug", type=str, default='', help="Augmentation type to include. Values: all/bt/c-mlm/rc/rephrasing/quad2text_trained/rephrasing+quad2text")

args = parser.parse_args()
print(f'Task Running for model: {args.model}, batch size: {args.batch}, gradient accumulation steps: {args.acc_steps}, using lora: {args.use_lora}')


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
  DATASET_TRAIN = ROOT + os.sep +'data/new_data/atc/roabsa_train_aug_final.csv'
  DATASET_TEST = ROOT + os.sep +'data/new_data/atc/roabsa_test.csv'
  DATASET_VAL = ROOT + os.sep +'data/new_data/atc/roabsa_eval.csv'
  tag = args.model.split('/')[-1]
  version = str(args.version)
  MODEL_SAVE_PATH = ROOT + os.sep + f'models/new_models/atc_aug_{tag}_causal_v{version}.pt' 

  MODEL_PRETRAINED_ATE = args.model 
  PRE_TRAINED_TOKENIZER_NAME = args.model
  WANDB_INIT_NAME = args.model + '-aug-' + args.aug 
  
  MAX_SOURCE_LEN = 512
  MAX_TARGET_LEN = 32
  
  BATCH_SIZE = args.batch
  EPOCHS = args.epochs
  LR = [3e-5, 1e-4, 2e-4, 3e-4]
  LR_IDX = 0
  EPS = 1e-5
  
  gradient_accumulation_steps = args.acc_steps
  
  USE_LABEL_SMOOTHING = False
  USE_GRADIENT_ACC = False
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
    do_sample=False,
    num_beams=30,
    length_penalty=0,
    repetition_penalty=1.2,
    num_return_sequences=1,
    no_repeat_ngram_size=5,
)

wandb.init(project="final_new_roabsa_atc", name=config.WANDB_INIT_NAME,
            config={
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
                    "Generate beams": gen_config.num_beams,
                    "Generate length penalty": gen_config.length_penalty,
                    "Generate repetition_penalty": gen_config.repetition_penalty,
                    "Generate no_repeat_ngram_size": gen_config.no_repeat_ngram_size,
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

df_train['text_cleaned'] = df_train['text_cleaned'].apply(remove_prompt)
df_val['text_cleaned'] = df_val['text_cleaned'].apply(remove_prompt)
df_test['text_cleaned'] = df_test['text_cleaned'].apply(remove_prompt)

if args.aug and args.aug!='all':
    df_train = select_train_instances(df_train)

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


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def remove_repeated_words(text):
  words = text.split()
  return " ".join(sorted(set(words), key=words.index))

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

tokenizer.add_eos_token=False  # only for mistral

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
                                ) 
train_dataset = Dataset.from_pandas(df_train)
val_dataset = Dataset.from_pandas(df_val)


def format_as_chat_with_output(example):
    messages =  [  
            {"role": "system", "content": "You are an expert linguist specialized in extracting categories of aspect terms identified in reviews."},
            {"role": "user", "content": example["text_cleaned"]},  # User input
            {"role": "assistant", "content": example["all_categories"]},  # Model response
        ]
    formatted_text = tokenizer.apply_chat_template(messages, 
                                                    tokenize=False, 
                                                    add_special_tokens=False,
                                                    add_generation_prompt=False)
    
    return formatted_text

def format_as_chat_without_output(text_cleaned, add_system_msg=True):
    if add_system_msg:
        system_message = {
            "role": "system",
            "content": "You are an expert linguist specialized in extracting categories of aspect terms identified in reviews."
        }
        messages =  [system_message, {"role": "user", "content": text_cleaned}] 
    else:
        messages =  [ {"role": "user", "content": text_cleaned}] 
        
    if hasattr(tokenizer, "apply_chat_template"):
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        if add_system_msg:
            formatted_text = (
                        f"System: {system_message['content']}\n"
                        f"User: {text_cleaned}\n"
                        f"###assistant:"
                )
        else:
            formatted_text = f"User: {text_cleaned}\n###assitant: "
    
    return formatted_text

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        return {
            'all_categories': item['all_categories'],
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
    refs = [ex["all_categories"] for ex in batch]

    padded = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_masks},
        padding=True,          # pad to max length in this batch
        return_tensors="pt"
    )
    return {
        "all_categories": refs,
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"]
    }


def evaluate_model(df_test, model, tokenizer):
  bleu_metric = evaluate.load("bleu")
  rouge_metric = evaluate.load("rouge")
  
  df_test['text_formatted'] = df_test['text_cleaned'].apply(format_as_chat_without_output)
  tokenized_outputs = tokenizer(df_test["text_formatted"].tolist(),
                                padding=True,  
                                truncation=True,
                                return_tensors="pt")
  df_test["input_ids"] = tokenized_outputs["input_ids"].tolist()
  df_test["attention_mask"] = tokenized_outputs["attention_mask"].tolist()
  
  test_dataloader = DataLoader(TestDataset(df_test), 
                             batch_size=config.BATCH_SIZE,
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
    references = batch['all_categories']
    
    with torch.no_grad():
      outputs = model.generate(input_ids=input_ids,
                               attention_mask= attention_mask,
                                max_new_tokens = config.MAX_TARGET_LEN,
                                return_dict_in_generate=True,
                                output_scores=True,
                                num_beams=30,
                                length_penalty=0,
                                repetition_penalty=1.2,
                                num_return_sequences=1,
                                no_repeat_ngram_size=5,
                                pad_token_id = tokenizer.pad_token_id
                               )  
      
    predictions = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

    cleaned_outputs = []
    for text in predictions:
        response_start = text.find("assistant")
        if response_start != -1:
            text = text[response_start + len("assistant"):].strip()
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
    early_stopping_patience=3,  # Stop training if no improvement after 3 evals
    early_stopping_threshold=0.01  # Require at least 0.01 improvement in eval_loss
)


training_args = TrainingArguments(
        output_dir=f"logs_{config.WANDB_INIT_NAME}",
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.LR[config.LR_IDX],
        logging_steps=10,
        num_train_epochs=config.EPOCHS,
        optim="adamw_8bit",
        report_to="wandb",
        run_name=config.WANDB_INIT_NAME,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1,
        warmup_ratio=0.1,
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
        dataloader_num_workers=128,
        # use_liger_kernel=True #check
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