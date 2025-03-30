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
login(token="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")

nltk.download('punkt')
torch.set_warn_always(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
  
  MODEL_SAVE_PATH = ROOT + os.sep +'models/new_models/atc_rollama3.1-8b-instruct_noBnBcConfig_v3.pt' 
  MODEL_PRETRAINED_ATE ='OpenLLM-Ro/RoLlama3.1-8b-Instruct' 
  PRE_TRAINED_TOKENIZER_NAME ='OpenLLM-Ro/RoLlama3.1-8b-Instruct'
  WANDB_INIT_NAME = "OpenLLM-Ro/RoLlama3.1-8b-Instruct-test"
  
  MAX_SOURCE_LEN = 512
  MAX_TARGET_LEN = 32
  
  BATCH_SIZE = 2
  EPOCHS = 6
  LR = [3e-5, 1e-4, 2e-4, 3e-4]
  LR_IDX = 0
  EPS = 1e-5
  
  gradient_accumulation_steps = 8
  label_smoothing = 0.1 # 0.0 if we do not use label smoothing for lowering the prediction confidence.
  
  USE_LABEL_SMOOTHING = False
  USE_GRADIENT_ACC = False
  USE_LORA = True
  USE_CONSTANT_SCHEDULER = False

use_cpu = False
resized_embeddings = False
# gen_config = GenerationConfig(
#     max_new_tokens = config.MAX_TARGET_LEN,
#     return_dict_in_generate=True,
#     output_scores=True,
#     do_sample=True,
#     num_beams=4,
#     temperature=0.1,
#     num_return_sequences=1,
#     no_repeat_ngram_size=4,
# )

gen_config = GenerationConfig(
    max_new_tokens = config.MAX_TARGET_LEN,
    return_dict_in_generate=True,
    output_scores=True,
    do_sample=False,
    num_beams=30,
    length_penalty=0,
    # do_sample=True,
    # temperature=0.1,
    # top_k=50,
    # top_p=0.9,
    repetition_penalty=1.2,
    num_return_sequences=1,
    no_repeat_ngram_size=5,
)

wandb.init(project="new_roabsa_atc", name=config.WANDB_INIT_NAME,
            config={
                    "Target": "all categories",
                    "Description": "Train ATC model with output pairs: <ATC1 is Pol1; ...;ATCn is PolN>",
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
                    # "Generate temperature": gen_config.temperature,
                    # "Generate top_p": gen_config.top_p,
                    # "Generate top_k": gen_config.top_k,
                    "Generate beams": gen_config.num_beams,
                    "Generate length penalty": gen_config.length_penalty,
                    "Generate repetition_penalty": gen_config.repetition_penalty,
                    "Generate no_repeat_ngram_size": gen_config.no_repeat_ngram_size,
                })

set_seed(config.SEED)
df_train = pd.read_csv(config.DATASET_TRAIN)
df_test = pd.read_csv(config.DATASET_TEST)
df_val = pd.read_csv(config.DATASET_VAL)

# df_train=df_train[:500]
# df_val=df_val[:100]
# df_test=df_test[:10]

print('DF TRAIN:', len(df_train))
print('DF TEST:', len(df_test))
print('DF VAL:', len(df_val))

df_train.dropna(subset=['text_cleaned'],inplace=True)
df_test.dropna(subset=['text_cleaned'],inplace=True)
df_val.dropna(subset=['text_cleaned'],inplace=True)

punctuation = string.punctuation
punctuation = re.sub('-','',punctuation)


# def get_batch_logps(
#     logits: "torch.Tensor", labels: "torch.Tensor", label_pad_token_id: int = IGNORE_INDEX
# ) -> Tuple["torch.Tensor", "torch.Tensor"]:
#     r"""
#     Computes the log probabilities of the given labels under the given logits.

#     Returns:
#         logps: A tensor of shape (batch_size,) containing the sum of log probabilities.
#         valid_length: A tensor of shape (batch_size,) containing the number of non-masked tokens.
#     """
#     if logits.shape[:-1] != labels.shape:
#         raise ValueError("Logits (batchsize x seqlen) and labels must have the same shape.")

#     labels = labels[:, 1:].clone()
#     logits = logits[:, :-1, :]
#     loss_mask = labels != label_pad_token_id
#     labels[labels == label_pad_token_id] = 0  # dummy token
#     per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
#     return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)


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
                                            token='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',
                                            use_cpu= False)

tokenizer.pad_token_id = 128002
model.generation_config.pad_token_id = tokenizer.pad_token_id

print("PAD Token:", tokenizer.pad_token)
print("PAD Token ID:", tokenizer.pad_token_id)
print("EOS Token:", tokenizer.eos_token)
print("EOS Token ID:", tokenizer.eos_token_id)
print("Padding Side:", tokenizer.padding_side)

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
                                ) # automatically pads sequences inside a batch to the longest one.

train_dataset = Dataset.from_pandas(df_train)
val_dataset = Dataset.from_pandas(df_val)


def format_as_chat_with_output(example):
    messages =  [  
            {"role": "system", "content": "You are an expert linguist specialized in extracting categories of aspect terms identified in reviews."},
            {"role": "user", "content": example["text_cleaned"]},  # User input
            {"role": "assistant", "content": example["all_categories_old"]},  # Model response
        ]
    formatted_text = tokenizer.apply_chat_template(messages, 
                                                    tokenize=False, 
                                                    # max_length=config.MAX_SOURCE_LEN,
                                                    add_special_tokens=False,
                                                    add_generation_prompt=False)
    
    return formatted_text

def format_as_chat_without_output(text_cleaned, add_system_msg=True):
    if add_system_msg:
        system_message = {
            "role": "system",
            "content": "You are an expert linguist specialized in extracting categories of aspect terms identified in reviews."
            # "content": (
            #     "You are an expert linguist specialized in extracting categories of aspect terms identified in reviews."
            #     """Given a user's review, list all aspects categories that have corresponding opinions.
            #     Review: <<I love how organized the store is, but the delivery was late.>>
            #     Aspects: store organization; delivery"""
            # )
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
            'all_categories_old': item['all_categories_old'],
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
    # `batch` is a list of dictionaries returned by `__getitem__` from dataset
    input_ids = [ex["input_ids"] for ex in batch]
    attention_masks = [ex["attention_mask"] for ex in batch]
    refs = [ex["all_categories_old"] for ex in batch]

    padded = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_masks},
        padding=True,          # pad to max length in this batch
        return_tensors="pt"
    )
    return {
        "all_categories_old": refs,
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"]
    }


def evaluate_model(df_test, model, tokenizer):
  bleu_metric = evaluate.load("bleu")
  rouge_metric = evaluate.load("rouge")
  
  df_test['text_formatted'] = df_test['text_cleaned'].apply(format_as_chat_without_output)
  tokenized_outputs = tokenizer(df_test["text_formatted"].tolist(),
                                padding=True,  # Ensure batch-level padding
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
    references = batch['all_categories_old']
    
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
                               ) # GenerateBeamDecoderOnlyOutput
      
    # do_sample=True,
    # temperature=0.1,
    # top_k=50,
    # top_p=0.9,
    # contains:
    #   outputs.sequences        -> the generated token IDs
    #   outputs.sequences_scores -> log probability scores for each sequence
    #   outputs.scores           -> a list of token-level logit distributions
    predictions = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    # log_probs = outputs.sequences_scores  # shape: [batch_size * num_return_sequences]. Values are in log space. Higher value is better
    # try:
    #     probs = torch.exp(log_probs)
    # except:
    #     probs = log_probs
    # print('Probs:',probs)

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

    # print('Start training...')
    # trainer.train()
    # print('Training finished')
    # trainer.save_model(config.MODEL_SAVE_PATH)
    # print(f'Model saved to: {config.MODEL_SAVE_PATH}')

    # if training_args.load_best_model_at_end:
    #     best_model_path = trainer.state.best_model_checkpoint  
    #     print(f"Loading the best model from checkpoint... {best_model_path}")
    #     if best_model_path is not None:
    #         trainer.model = trainer.model.from_pretrained(best_model_path)  
    # else:
    #     print("Warning: No best model checkpoint found. Using the last trained model.")
    
    base_model = model.from_pretrained(config.MODEL_SAVE_PATH)
    trainer.model = PeftModel.from_pretrained(base_model, '/export/home/acs/stud/a/alina.gheorghe2505/ssl/logs_OpenLLM-Ro/RoLlama3.1-8b-Instruct/checkpoint-966')   
     
    print('Testing the model...')
    evaluate_model(df_test=df_test,
                   model=trainer.model,
                   tokenizer=tokenizer)
    print('Evaluation finished')

except Exception as e:
  print(e)