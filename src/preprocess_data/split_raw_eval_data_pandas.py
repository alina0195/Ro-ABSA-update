import pandas as pd
import re, os
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

BASE_DIR = Path(__file__).parent

class Config:
    SEED=42
    IN_DIR = str(BASE_DIR) + os.sep + 'raw_data'
    OUT_DIR = str(BASE_DIR) + os.sep + 'processed_data'
        
    FILE_NAME_TRAIN_AUG = 'df_train_augmented_almostRare.csv'
    FILE_NAME_TRAIN_AUG_RARE = 'df_train_augmented_rare.csv'
    FILE_NAME_TRAIN_MANUAL = 'df_train_manual.csv'
    
    FILE_NAME = 'test_eval_roabsa_new_jsonmin.json'
    FILE_NAME_INCONSISTENCES = 'roabsa_test_eval_inconsistences.csv'
    FILE_NAME_EVAL_CSV = 'roabsa_eval.csv'
    FILE_NAME_TEST_CSV = 'roabsa_test.csv'
    
    COLS_TO_REMOVE =['lead_time','updated_at','created_at',
                    'annotation_id','sents_length','rating',
                    'tokenized_sents']


df = pd.read_json(Config.IN_DIR+os.sep+Config.FILE_NAME)
print(df)
print(df.columns)
print(df.iloc[0])

polarity_map ={
    'positive':'Positive',
    'negative':'Negative',
    'neutral':'Neutral',
}

def remove_missing_data(df):
    print('Initial len:', len(df))
    df = df.drop(columns=Config.COLS_TO_REMOVE)
    df = df.dropna()
    df = df.dropna(subset=['aspects_polarities'])
    df = df.dropna(subset=['body'])
    print('Final len:', len(df))
    return df

def transform_aspects_polarities(aspects_polarities):
    # get values for all_categories, all_ate and all_polarities  columns
    if type(aspects_polarities)==dict:
        unique_categories =[]
        all_categories =[]
        all_polarities =[]
        pairs = aspects_polarities['choices']
        for pair in pairs:
            category = pair.split('-')[0].strip()
            polarity = polarity_map[pair.split('-')[1].strip()]
            
            if category not in unique_categories:
                unique_categories.append(category)
            
            all_categories.append(category)
            all_polarities.append(polarity)
            
        all_categories='; '.join(all_categories) 
        unique_categories='; '.join(unique_categories)            
        all_polarities='; '.join(all_polarities)    
                
    else:
        all_categories=aspects_polarities.split('-')[0].strip()
        all_polarities=polarity_map[aspects_polarities.split('-')[1].strip()]
        unique_categories = all_categories
        
    return all_categories, unique_categories, all_polarities

def check_for_exceptions(df:pd.DataFrame)->pd.DataFrame:
    for _, row in df.iterrows():
        exceptions = []
        
        polarities_len = len(row['all_polarities'].split(';')) 
        categoies_len = len(row['all_categories_old'].split(';'))
        
        if polarities_len!=categoies_len:
            exceptions.append(row['id'])
    
    print('Exceptions found:')
    print(exceptions)
    
    if len(exceptions)>0:
        df = df[~df['id'].isin(exceptions)]
    print('After dropping exception pairs df len:', len(df))
    
    return df

def clean_up_review(review):
    def remove_emojis(text):
        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002500-\U00002BEF"  # chinese char
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642"
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  # dingbats
                                u"\u3030"
                                "]+", flags=re.UNICODE)

        # Replace emojis with an empty string
        text_without_emojis = emoji_pattern.sub(r'', text)

        return text_without_emojis

    try:
      # Remove urls, bullet points, extra whitespace and HTML tags
      review = remove_emojis(review)
      review = re.sub(r'https?://\S+|www\.\S+', '', review)
      review = re.sub(r'<.*?>', ' ', review)
      review = re.sub(r'(\s*[\*\•]\s*)', ' ', review)
      review = re.sub(r'\s+', ' ', review).strip()

      # Replace special characters with Romanian diacritics
      review = review.replace('&acirc;', 'â').replace('&icirc;', 'î').replace('&scedil;', 'ș').replace('&tcedil;', 'ț').replace('&Acirc;', 'Â').replace('&Icirc;', 'Î').replace('&Scedil;', 'Ș').replace('&Tcedil;', 'Ț')
      review = review.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
      return review

    except Exception as e:
      print(f"Error {e} occured for:\n{review}")
      return None

def add_specific_columns(df):
    # add text_cleaned,  data_origin
    df['data_origin'] = 'manual'
    df['text_cleaned'] = df['text'].apply(lambda x: clean_up_review(x))
    return df

def get_all_categories_in_list(df, col_name):
    all_categ = []
    for _, row in df.iterrows():
        try:
            cat = row[col_name].split(';')
            cat = [c.strip() for c in cat]
            all_categ.extend(cat)
        except Exception as e:
            print(row)
            print(f'Instance {row[col_name]} raised error')
    return all_categ

def plot_frequencies(freq:dict, title):
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.title(title)
    
    fig, ax = plt.subplots()
    ax.set_ylim(bottom=0, top=5000)
    ax.set_xticks(range(len(freq)))
    ax.set_xticklabels(labels=freq.keys(), rotation=90)
    ax.set_title(title)
    ax.bar(freq.keys(), freq.values())
        
    df_name = title.split(' ')[0] +'_'+ title.split('<<')[1].split('>>')[0].split('.')[0] + '.jpg'
    print('Df name:', df_name)
    fig.savefig(Config.OUT_DIR+ os.sep + 'stats'+ os.sep + df_name)
    print('Plot saved')

def show_statistics(df: pd.DataFrame, filename:str, include_all_categories:bool=False):
    print(f'Show statistics for train dataset <<{filename}>>:\n')
    print('Initial len:', len(df))
    df.drop_duplicates(subset=['text_cleaned'], inplace=True)
    print('No duplicates len:', len(df))
    df.dropna(subset=['all_categories','all_polarities','text_cleaned'],inplace=True)

    if include_all_categories:
        all_cats_old=get_all_categories_in_list(df, 'all_categories_old')
        freq_lbls_all = Counter(all_cats_old).most_common(len(all_cats_old))
        freq_lbls_all = {item[0]:item[1] for item in freq_lbls_all}
        print('\nFrequent original categories:')
        print(freq_lbls_all)
        plot_frequencies(freq_lbls_all, f"Initial categories distribution from <<{filename}>>")
        
    all_cats=get_all_categories_in_list(df, 'all_categories')
    freq_lbls_unique = Counter(all_cats).most_common(len(all_cats))
    freq_lbls_unique = {item[0]:item[1] for item in freq_lbls_unique}
    print('\nFrequent unique categories:')
    print(freq_lbls_unique)
    plot_frequencies(freq_lbls_unique, f"Unique categories distribution from <<{filename}>>")
    
    print('\nShow data origin distribution:')
    origin_freq = Counter(df['data_origin'].values).most_common()
    print(origin_freq)
    

df = remove_missing_data(df)
df['all_categories'], df['all_categories_old'], df['all_polarities'] = zip(*df['aspects_polarities'].apply(lambda x: transform_aspects_polarities(x)))
print('Current df len:', len(df))

df = check_for_exceptions(df=df)
df.rename(columns={'body':'text'}, inplace=True)

df_22=df[df['annotator']==22]
df_26=df[df['annotator']==26]
df_22.drop(columns=['annotator','aspects_polarities'], inplace=True)
df_26.drop(columns=['annotator','aspects_polarities'], inplace=True)

print('df 22 len:',len(df_22))
print('df 26 len:',len(df_26))


print('Checking for inconsistencies:')
df_concat = pd.concat([df_22,df_26], axis=0).drop_duplicates(keep=False)
print('Nb of inconsistencies:', len(df_concat))
df_concat.to_csv(Config.OUT_DIR + os.sep + Config.FILE_NAME_INCONSISTENCES, index=False)

# Select only one annotator's values. Currently there are only 12 instances with different labels 
df = df_22
df = add_specific_columns(df)
df.drop_duplicates(subset=['text_cleaned','all_categories_old'], inplace=True)
print('Df len with no duplicates:',len(df))
# print(df.columns)
# print(df.iloc[0])

# Check for common values with training dataset
print('-'*20)
df_train = pd.read_csv(Config.OUT_DIR + os.sep +'train'+ os.sep + Config.FILE_NAME_TRAIN_MANUAL)
show_statistics(df_train, Config.FILE_NAME_TRAIN_MANUAL )

print('-'*20)
df_train_aug_rare_processed = pd.read_csv(Config.OUT_DIR + os.sep + 'train'+os.sep + Config.FILE_NAME_TRAIN_AUG_RARE)
show_statistics(df_train_aug_rare_processed,Config.FILE_NAME_TRAIN_AUG_RARE )

print('-'*20)
df_train_aug_processed = pd.read_csv(Config.OUT_DIR + os.sep + 'train' + os.sep + Config.FILE_NAME_TRAIN_AUG)
show_statistics(df_train_aug_processed, Config.FILE_NAME_TRAIN_AUG)

df_concat = pd.concat([df[['all_categories','all_polarities','text']], df_train[['all_categories','all_polarities','text']]], axis=0).drop_duplicates(keep=False)
print('len df train:', len(df_train),'\nlen df_test_eval:',len(df),'\nlen df concat:',len(df_concat))
assert len(df_train)+len(df)==len(df_concat)
print(df_concat)

df_test, df_val = train_test_split(df, test_size=0.50, random_state=Config.SEED)
print('Test dataset size:',len(df_test),'\nValidation dataset size:', len(df_val))
print('-'*20)
show_statistics(df_test, Config.FILE_NAME_TEST_CSV)
show_statistics(df_val, Config.FILE_NAME_EVAL_CSV)

df_test.to_csv(Config.OUT_DIR + os.sep + Config.FILE_NAME_TEST_CSV, index=False)
df_val.to_csv(Config.OUT_DIR + os.sep + Config.FILE_NAME_EVAL_CSV, index=False)
print(f'Datasets successfully saved to {Config.OUT_DIR}')