import pandas as pd
from collections import Counter 


path = 'processed_data\\good_to_use_data_absa\\train_absaPairs.csv'
df = pd.read_csv(path)


def get_all_independent_categories(df):
    all_categ = []
    for _, row in df.iterrows():
        cat = row['absa_target'].split(';')
        cat = [c.strip() for c in cat]
        all_categ.extend(cat)
    return all_categ

all_categ = get_all_independent_categories(df)
freq_categories = Counter(all_categ)
freq_labels = Counter(df['absa_target'])

categories_dist = [item[0] for item in freq_categories.most_common(len(freq_categories))]   # 
print('Absa targets distribution:', categories_dist)

rare_categories = categories_dist[-30:]
print('Rare absa targets distribution:', rare_categories)


def to_keep_rare(row):
    labels = row['absa_target'].split(';')
    labels = [lbl.strip() for lbl in labels]
    
    for rare_cat in rare_categories:
        if rare_cat in labels:
            return 'add'
    return 'not_add'

max_len = 100

def is_short(row):
    text_len = len(row['text_cleaned'].split(' '))
    if text_len < max_len:
        return 'short'
    else:
        return 'not_short'        

def has_common_categories(target1, target2):
    categories_from_first_row = []
    for pair in target1.split(';'):
        categories_from_first_row.append(pair.split(' is ')[0].strip())
    
    categories_from_second_row = []
    for pair in target2.split(';'):
        categories_from_second_row.append(pair.split(' is ')[0].strip())

    is_unique = False
    
    for category in categories_from_second_row:
        if category in categories_from_first_row:
            is_unique = True
            print('First label:',target1)
            print('Second label:', target2)
            print('Found:', category)
            return is_unique
    
    return is_unique
    
def next_non_overlapping(first_row, df, i):
    """Return the index of the first row after i whose categories don't overlap."""
    for j in range(i+1, len(df)):
        if not has_common_categories(first_row['absa_target'], df.iloc[j]['absa_target']):
            return j
    return None        # not found

    
def get_concat_instances(df):
    new_rows = []
    for i in range(len(df)):
        first_row = df.iloc[i]
        next_index = next_non_overlapping(first_row, df, i)
        if next_index:
            second_row = df.iloc[next_index]
        else:
            print('No available index. Pass')
            continue
        new_row = {}
        new_row['id'] = first_row['id'] + second_row['id']
        new_row['text_cleaned'] = first_row['text_cleaned']+'. ' + second_row['text_cleaned']
        new_row['text'] = first_row['text']+'. ' + second_row['text']
        new_row['absa_input'] = first_row['absa_input'].split('</s>')[0].strip() + '. ' + second_row['text_cleaned'] + '</s>'
        new_row['absa_target'] = first_row['absa_target'] +'; ' + second_row['absa_target']
        
        new_row['data_origin'] = 'random-concatenation'
        new_rows.append(new_row)
    df_to_add = pd.DataFrame(new_rows)
    df_to_add['keep'] = 'add'
    df_to_add['short'] = 'not_short'
    return df_to_add


df['keep'] = df.apply(lambda x: to_keep_rare(x), axis=1)
df['short'] = df.apply(lambda x: is_short(x), axis=1)

df_selected_for_concat = df[(df['keep']=='add') & (df['short']=='short')]
print('DF selected for concatenation:', len(df_selected_for_concat))
df_with_reviews_concatenated = get_concat_instances(df_selected_for_concat)
df_with_reviews_concatenated.drop_duplicates(subset=['text_cleaned'], inplace=True)

df_rare_lbls = df[(df['keep']=='add')]

df.drop(columns=['keep','short'], inplace=True)
df_rare_lbls.drop(columns=['keep','short'], inplace=True)
df_with_reviews_concatenated.drop(columns=['keep','short'], inplace=True)


print(f'df rare lbls columns: {df_rare_lbls.columns}')
print(f'df concat columns: {df_with_reviews_concatenated.columns}')
print(f'df original columns: {df.columns}')

df_and_concatenated_reviews = pd.concat([df, df_with_reviews_concatenated], axis=0, ignore_index=True)
df_rare_lbls.to_csv('df_with_only_rare_lbls.csv',index=False)
df_and_concatenated_reviews.to_csv('df_train-augmented_and_rc.csv',index=False)

print('DF original:', len(df))
print('DF rare lbls:', len(df_rare_lbls))
print('DF with only concatenated reviews:', len(df_with_reviews_concatenated))
print('DF original and reviews random concatenated:', len(df_and_concatenated_reviews))

print('\nDistribution of original data before adding random concatenated reviews:')
all_categ_before = get_all_independent_categories(df)
freq_categories_before = Counter(all_categ_before).most_common()
print(freq_categories_before)
print('\nDistribution of original data after adding random concatenated reviews:')
all_categ_after = get_all_independent_categories(df_and_concatenated_reviews)
freq_categories_after = Counter(all_categ_after).most_common()
print(freq_categories_after)

