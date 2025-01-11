import pandas as pd
import os 
from pathlib import Path

BASE_DIR = Path(__file__).parent

class Config:
    SEED=42
    IN_DIR = str(BASE_DIR) + os.sep + 'processed_data' + os.sep + 'good_to_use_data_atc'
    OUT_DIR = str(BASE_DIR) + os.sep + 'processed_data' + os.sep + 'good_to_use_data_absa'
    
    TRAIN_FILE = IN_DIR+ os.sep + 'roabsa_train.csv'
    TEST_FILE = IN_DIR + os.sep + 'roabsa_test.csv'
    VAL_FILE = IN_DIR + os.sep + 'roabsa_eval.csv'
    PROMPT_ABSA = 'Extract pairs of aspect categories with their corresponding opinions from the following Romanian review: '
    

def create_df_absa_in_out(df, filename_save):
    new_df = pd.DataFrame(columns=df.columns)
    cnt = 0
    for _, row in df.iterrows():
        categories = row['all_categories_old']
        polarities = row['all_polarities']
        
        categories = [cat.strip() for cat in categories.split(';')]
        polarities = [pol.strip().lower() for pol in polarities.split(';')]
        target = []
        
        new_row = row.copy()
        new_row['absa_input'] = Config.PROMPT_ABSA + new_row['text_cleaned'] + ' </s>'
        
        for cat, pol in zip(categories, polarities):
            current_target = cat + ' is ' + pol
            target.append(current_target)
            
        target = '; '.join(target) 
        new_row['absa_target'] = target 
        
        new_df.loc[cnt] = new_row
        cnt += 1

    new_df.drop(['all_categories','all_categories_old','all_polarities'], inplace=True, axis=1)
    new_df.to_csv(filename_save, index=False)
    return new_df


def apply_transformation(file_path:str, split_name:str):
    df = pd.read_csv(file_path)
    df['absa_input']= ''
    df['absa_target']= ''
    
    file_save = Config.OUT_DIR + os.sep  + f'{split_name}_absaPairs.csv'
    create_df_absa_in_out(df, file_save)
    print(f'Successfully applied for {file_path}. New data saved in: {file_save}')


if __name__=='__main__':
    apply_transformation(Config.TRAIN_FILE,'train')
    # apply_transformation(Config.TEST_FILE, 'test')
    # apply_transformation(Config.VAL_FILE, 'eval')
    
