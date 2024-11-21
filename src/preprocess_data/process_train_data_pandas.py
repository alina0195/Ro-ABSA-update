import pandas as pd
import re, os
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent

class Config:
    SEED=42
    IN_DIR = str(BASE_DIR) + os.sep + 'raw_data'
    OUT_DIR = str(BASE_DIR) + os.sep + 'processed_data'
        
    # FILE_NAME_TRAIN_MANUAL = os.sep + 'old_train' + os.sep +'df_train_manual.csv'
    FILE_NAME_TRAIN_MANUAL = os.sep + 'old_train' + os.sep +'df_train_augmented_almostRare.csv'
    # FILE_NAME_TRAIN_MANUAL = os.sep + 'old_train' + os.sep +'df_train_augmented_rare.csv'
    
    FILE_NAME = 'train_roabsa_new_jsonmin.json'
    FILE_NAME_INCONSISTENCES = 'roabsa_train_inconsistences.csv'
    
    FILE_NAME_SAVE = 'roabsa_train_augm_almostRare_new.csv'
    FILE_NAME_SAVE_PROCESSED = 'roabsa_train_augm_almostRare_new_process.csv'
    
    COLS_TO_REMOVE =['lead_time','updated_at','created_at',
                    'annotation_id','sents_length','rating',
                    'tokenized_sents']


df = pd.read_json(Config.IN_DIR + os.sep + Config.FILE_NAME)
df_train_old = pd.read_csv(Config.OUT_DIR + os.sep+Config.FILE_NAME_TRAIN_MANUAL)
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
    print(f'Show statistics for dataset <<{filename}>>:\n')
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
    

def get_corresponding_new_labels(row, df_new:pd.DataFrame, df_old:pd.DataFrame):
    current_text = row['text_cleaned']
    
    if len(current_text)==0:
        raise ValueError(row)

    if type(current_text)!=str:
        current_text = current_text.values[0]
        print(current_text)
    try:
        if current_text in df_new['text_cleaned'].values:
            text = current_text
        else:
            # this is and augmented review 
            # find the instance with the same id and with data_origin =='manual' 
            rows_with_same_id = df_old[df_old['id']==row['id']]
            
            text = rows_with_same_id[rows_with_same_id['data_origin']=='manual']['text_cleaned']
            if len(text)==0:
                print('No text found. Ids were:', rows_with_same_id[row['id']])
    except Exception as e:
        print('Exception when getting the original text by id:', str(e))
        
    # find new labels from new dataframe based on the text
    try:
        if type(text)!=str:
            print('HERE1:', text)
            text = text.values[0] 
            print('HERE2:', text)
    except Exception as e:
        print(f'Exception for text: {text}', str(e))
        
    try:
        if text  in df_new['text_cleaned'].values:
            corresponding_all_categories = df_new[df_new['text_cleaned']==text]['all_categories_old'].values[0] 
            corresponding_unique_categories = df_new[df_new['text_cleaned']==text]['all_categories'].values[0] 
            corresponding_polarities = df_new[df_new['text_cleaned']==text]['all_polarities'].values[0]
        else:
            corresponding_all_categories = '' 
            corresponding_unique_categories = '' 
            corresponding_polarities = '' 
    except Exception as e:
        print(f'Exception for text when getting corresponding categories: {text}.', str(e))
        corresponding_all_categories = '' 
        corresponding_unique_categories = '' 
        corresponding_polarities = '' 
    
    return corresponding_all_categories, corresponding_unique_categories, corresponding_polarities


df = remove_missing_data(df)
df['all_categories'], df['all_categories_old'], df['all_polarities'] = zip(*df['aspects_polarities'].apply(lambda x: transform_aspects_polarities(x)))
print('Current df len:', len(df))

df = check_for_exceptions(df=df)
df.rename(columns={'body':'text'}, inplace=True)
print('Df len before dropping duplicated:', len(df))
df.drop_duplicates(subset=['text','all_categories_old'], inplace=True)
print('Df len after dropping duplicated:', len(df))
df = add_specific_columns(df)


print('Checking for different reviews between old and new training data:')
df_concat = pd.concat([df_train_old['text_cleaned'],df['text_cleaned']], axis=0).drop_duplicates(keep=False)
mask = df['text_cleaned'].isin(df_concat) 
df_incons = df[mask]
print(df_incons)
print(len(df_incons))
print(df_incons.iloc[0])
df_incons = add_specific_columns(df_incons)
df_incons.drop(columns=['annotator'], inplace=True)
df_incons.to_csv(Config.OUT_DIR + os.sep + Config.FILE_NAME_INCONSISTENCES, index=False)

df.drop_duplicates(subset=['text_cleaned','all_categories_old'], inplace=True)
print('Df len with no duplicates:',len(df))

df.to_csv(Config.OUT_DIR + os.sep + Config.FILE_NAME_SAVE, index=False)
df.drop(columns=['annotator'], inplace=True)
print(f'Dataset successfully saved to {Config.OUT_DIR}')

# Check for different labels between the old and the new training dataset
print('-'*20)
show_statistics(df, Config.FILE_NAME_SAVE)

mask = df['text'].isin(df_incons['text'])
df_new = df[~mask] 
print('Df new without new reviews:')
print(df_new)

df_train_old['all_categories_new'], df_train_old['all_categories_old_new'], df_train_old['all_polarities_new'] = zip(*df_train_old.apply(lambda x: get_corresponding_new_labels(row=x,
                                                                                                                                                                                df_new=df_new,
                                                                                                                                                                                df_old=df_train_old), axis=1))

print('Result:')
print(df_train_old[['all_categories_new','all_categories']])
print('*'*20)
print(df_train_old[['all_polarities_new','all_polarities']])

df_incons['all_categories_new'] = df_incons['all_categories']
df_incons['all_categories_old_new'] = df_incons['all_categories_old']
df_incons['all_polarities_new'] = df_incons['all_polarities']

df_train_old_and_new = pd.concat([df_train_old,df_incons], axis=0, ignore_index=True)
print('Len final:', len(df_train_old_and_new))
df_train_old_and_new.to_csv(Config.OUT_DIR + os.sep + Config.FILE_NAME_SAVE_PROCESSED)
