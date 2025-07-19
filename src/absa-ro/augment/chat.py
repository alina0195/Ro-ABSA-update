import config
import requests
import json
import torch
from transformers import  GenerationConfig
import torch.nn.functional as F

class ChatContextCreator():
    def __init__(self, config: config.ChatCompletionConfig):
        self.config = config
        self.model = config.model.strip()
        
        self.system_prompt = self.config.system_prompt.strip()
        self.user_prompt = self.config.user_prompt.strip()
        
        self.messages = []
        self.messages.append({"role":"system","content":self.system_prompt})
        
    
    def assistant(self, content: str):
      return { "role": "assistant", "content": content }

    def user(self, content: str):
        return { "role": "user", "content": content }

    def add_messages(self, user_content:str='', assistant_content:str=''):
        if len(user_content)>0:
            self.messages.append(self.user(user_content))
        if len(assistant_content)>0:
            self.messages.append(self.assistant(assistant_content))

    def update_user_messages(self, value):
        self.messages[1]["content"]=value

    def clear_messages(self):
        self.messages = []
        self.messages.append({"role":"system","content":self.system_prompt})
        
        
    def chat_completion(self, 
                        messages: list[dict], 
                        temperature: float=None, 
                        top_p: float=None,
                        top_k: int=None,
                        no_repeat_ngram_size: int=None,
                        repetition_penalty: int=None,
                        max_tokens: int=None):

        if not temperature:
            temperature = self.config.temperature
        if not top_p:
            top_p = self.config.top_p
        if not max_tokens:
            max_tokens = self.config.max_tokens
        if not top_k:
            top_k = self.config.top_k
        if not repetition_penalty:
            repetition_penalty = self.config.repetition_penalty
        if not no_repeat_ngram_size:
            no_repeat_ngram_size = self.config.no_repeat_ngram_size
        
        data = {
            "model":self.model,
            "messages":messages,
            "temperature":temperature,
            "top_p":top_p,
            "top_k":top_k,
            "repetition_penalty":repetition_penalty,
            "no_repeat_ngram_size":no_repeat_ngram_size,
            "max_new_tokens": max_tokens
            }
        
        headers = {"Authorization": f"Bearer sk-xxx",
                    "Content-Type": "application/json"}
        response = requests.post("https://chat.readerbench.com/api/chat/completions",
                                headers=headers,
                                data=json.dumps(data))
        if response:
            response_text = json.loads(response.text)['choices'][0]['message']['content']
            print(response_text)
            return response_text
        else:
            print('No response received:', response)
            return ''
    
    def local_model_completion(self,
                                batch,
                                model, 
                                tokenizer,
                                temperature: float=None, 
                                top_p: float=None,
                                top_k: int=None,
                                do_sample: bool=None,
                                no_repeat_ngram_size: int=None,
                                repetition_penalty: float=None,
                                max_tokens: int=None):

        if not temperature:
            temperature = self.config.temperature
        if not top_p:
            top_p = self.config.top_p
        if not max_tokens:
            max_tokens = self.config.max_tokens
        if not top_k:
            top_k = self.config.top_k
        if not repetition_penalty:
            repetition_penalty = self.config.repetition_penalty
        if not no_repeat_ngram_size:
            no_repeat_ngram_size = self.config.no_repeat_ngram_size
        if not do_sample:
            do_sample = self.config.do_sample
               
        generation_config = GenerationConfig(eos_token_id= tokenizer.eos_token_id,  
                                            pad_token_id = tokenizer.pad_token_id,   
                                            use_cache = False,
                                            temperature = temperature,  
                                            max_new_tokens=max_tokens,
                                            top_p=top_p,
                                            top_k=top_k,
                                            no_repeat_ngram_size=no_repeat_ngram_size,
                                            repetition_penalty=repetition_penalty,
                                            do_sample = do_sample,
                                            return_dict_in_generate=True,
                                            output_scores=True,
                                            )
        
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     generation_config = generation_config
                                    ) 
            predictions = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

            cleaned_predictions = []
            for pred in predictions:
                idx = pred.rfind('assistant')
                if idx != -1:
                    cleaned_text = pred[idx + len('assistant'):]
                else:
                    cleaned_text = pred
                cleaned_predictions.append(cleaned_text.strip())
            print(f'{cleaned_predictions=}')
            del outputs, predictions
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  
            
        return cleaned_predictions   

    
    