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
                  TaskType, 
                  PeftModel,
                  prepare_model_for_kbit_training,
                  get_peft_model)           
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
parser.add_argument("--use_lora", type=bool, default=False, help="Use Lora or Not")
parser.add_argument("--epochs", type=int, default=10, help="Number of Epochs")
parser.add_argument("--version", type=int, default=1, help="Version for model's name saved")

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
  DATASET_TRAIN = ROOT + os.sep +'data/new_data/alsa/train_alsa.csv'
  DATASET_TEST = ROOT + os.sep +'data/new_data/alsa/test_alsa.csv'
  DATASET_VAL = ROOT + os.sep +'data/new_data/alsa/val_alsa.csv'
  tag = args.model.split('/')[-1]
  version = str(args.version)
  MODEL_SAVE_PATH = ROOT + os.sep +f'models/new_models/alsa_{tag}_v{args.version}.pt' 
  MODEL_PRETRAINED_ATE = args.model 
  PRE_TRAINED_TOKENIZER_NAME = args.model
  WANDB_INIT_NAME = args.model 
  
  MAX_SOURCE_LEN = 512
  MAX_TARGET_LEN = 10
  
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
      
  USE_LORA = args.use_lora
  USE_CONSTANT_SCHEDULER = True
  lora_config = LoraConfig(
                        r=16,
                        lora_alpha=32,
                        target_modules=["q", "v", "k"],
                        lora_dropout=0.1,
                        bias="none",
                        task_type=TaskType.SEQ_2_SEQ_LM,
                        use_rslora=True
                        )
use_cpu = False
resized_embeddings = False

gen_config = GenerationConfig(
    max_new_tokens = config.MAX_TARGET_LEN,
    return_dict_in_generate=True,
    output_scores=True,
    num_beams=3,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
)

wandb.init(project="new_roabsa_alsa", name=config.WANDB_INIT_NAME,
            config={
                      "learning rate": config.LR[config.LR_IDX],
                      "optimizer": "adam",
                      "epochs": config.EPOCHS,
                      "batch": config.BATCH_SIZE,
                      "pretrained model":config.MODEL_PRETRAINED_ATE,
                      "pretrained tokenizer":config.PRE_TRAINED_TOKENIZER_NAME,
                      "model save path": config.MODEL_SAVE_PATH,
                      "constant scheduler": config.USE_CONSTANT_SCHEDULER,
                      "lora": config.USE_LORA,
                      "MAX_SOURCE_LEN": config.MAX_SOURCE_LEN,
                      "MAX_TARGET_LEN": config.MAX_TARGET_LEN,
                      "dataset train": f"{config.DATASET_TRAIN}",
                      "dataset test": f"{config.DATASET_TEST}",
                      "dataset val": f"{config.DATASET_VAL}",
                      "Accelerate used": "False",
                      "Generate temperature": gen_config.temperature,
                      "Generate num_return_sequences":gen_config.num_return_sequences,
                      "Generate num_beams": gen_config.num_beams,
                      "Generate no_repeat_ngram_size": gen_config.no_repeat_ngram_size,
                      })
set_seed(config.SEED)

df_train = pd.read_csv(config.DATASET_TRAIN)
df_test = pd.read_csv(config.DATASET_TEST)
df_val = pd.read_csv(config.DATASET_VAL)

def f1(pred, target):
      return f1_score(target, pred, average='weighted')


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
                token=token,
                low_cpu_mem_usage=True if not use_cpu else False,
                )
    else:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                token=token,
                low_cpu_mem_usage=True if not use_cpu else False,
                )
        base_model = prepare_model_for_kbit_training(
                    base_model,
                    use_gradient_checkpointing=True   # keeps memory small
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
        model = get_peft_model(base_model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              use_fast=False,
                                              token=token)

    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(
                                            model_path=config.MODEL_PRETRAINED_ATE,
                                            load_in_8bits= True,
                                            token='xxx',
                                            use_cpu= False)

print("PAD Token:", tokenizer.pad_token)
print("PAD Token ID:", tokenizer.pad_token_id)
print("EOS Token:", tokenizer.eos_token)
print("EOS Token ID:", tokenizer.eos_token_id)
print("Padding Side:", tokenizer.padding_side)


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
        
class ALSADataset(Dataset):
    def __init__(self, df, tokenizer):
        self.inputs = df["alsa_input"].tolist()
        self.targets = df["polarity"].tolist()
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
 
    
train_dataset = ALSADataset(df_train, tokenizer)
val_dataset = ALSADataset(df_val, tokenizer)


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
    exact_match = evaluate.load("exact_match")

    test_dataset = ALSADataset(df_test, tokenizer)
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
    f1_scores, em_scores = [], []
    
    for batch in test_dataloader: 
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels']

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
                max_new_tokens=config.MAX_TARGET_LEN,
            )
        labels_for_decode = labels.clone()
        labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id

        decoded_preds = safe_batch_decode(tokenizer, outputs.sequences)
        decoded_preds = [p.strip().lower() for p in decoded_preds]
        decoded_labels = safe_batch_decode(tokenizer, labels)
        decoded_labels = [r.strip().lower() for r in decoded_labels]

        preds.extend(decoded_preds)
        refs.extend(decoded_labels)
        f1_val = f1(decoded_preds, decoded_labels)
        em = exact_match.compute(predictions=decoded_preds, references=decoded_labels)
        f1_scores.append(f1_val)
        em_scores.append(em['exact_match'])

        print("Reference:", decoded_labels)
        print("Prediction:", decoded_preds)
        
        
    avg_f1 = np.mean(f1_scores)
    avg_em = np.mean(em_scores)
    print("\Exact Match:", avg_em)
    print("F1:", avg_f1)

    wandb.log({
        "Test F1": avg_f1,
        "Test exact match": avg_em
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
        max_grad_norm=1.0, 
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