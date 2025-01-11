import pandas as pd
import os 
from pathlib import Path
from transformers import AutoTokenizer


BASE_DIR = Path(__file__).parent

class Config:
    SEED=42
    IN_DIR = str(BASE_DIR) + os.sep + 'processed_data' + os.sep + 'good_to_use_data_atc'
    OUT_DIR = str(BASE_DIR) + os.sep + 'processed_data' + os.sep + 'stats'
    
    TRAIN_FILE = IN_DIR+ os.sep + 'roabsa_train.csv'
    TEST_FILE = IN_DIR + os.sep + 'roabsa_test.csv'
    VAL_FILE = IN_DIR + os.sep + 'roabsa_eval.csv'
    

def get_tokens_nb(text:str, tokenizer:AutoTokenizer):
    encoded_dict = tokenizer.encode_plus(text,
                                        add_special_tokens=True,
                                        return_tensors='pt')
    return len(encoded_dict['input_ids'][0])


def get_tokens_nb_stats(tokenizer:AutoTokenizer, df_path:str, col_name:str, split_name:str, threshold:int):
    df = pd.read_csv(df_path)
    
    df['tokens_nb'] = df[col_name].apply(lambda x: get_tokens_nb(x,tokenizer))
    from matplotlib import pyplot as plt
    
    min_nb = df['tokens_nb'].min()
    max_nb = df['tokens_nb'].max()
    rows_over_threshold = df[df['tokens_nb'] > threshold]

    print('Reviews over 512 tokens:', len(rows_over_threshold))
    for rev in rows_over_threshold['text']:
        print(rev)
        print('-'*20)
    
    nb_over_threshold = len(rows_over_threshold)
    rows_over_threshold = rows_over_threshold['tokens_nb'].values
    rows_over_threshold = ', '.join([str(elem) for elem in rows_over_threshold])
    df['tokens_nb'].plot(kind='hist', bins=20, title=f'{split_name}: Tokens Length Distribution')
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.text(0.05, 0.95, f'Min: {min_nb:.2f}\nMax: {max_nb:.2f}\n#Over {threshold}: {nb_over_threshold}\n\nOver {threshold}: {rows_over_threshold}',
            transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    
    plt.legend()
    plt.savefig(Config.OUT_DIR + os.sep + f'tokens_histogram_{split_name}.png')
    plt.show()


tokenizer = AutoTokenizer.from_pretrained('bigscience/mt0-small')

if __name__=='__main__':
    get_tokens_nb_stats(tokenizer, Config.TRAIN_FILE ,'text_cleaned' ,'train',512)
    # get_tokens_nb_stats(tokenizer, Config.TEST_FILE ,'text_cleaned' ,'test',512)
    # get_tokens_nb_stats(tokenizer, Config.VAL_FILE ,'text_cleaned' ,'val', 512)
    