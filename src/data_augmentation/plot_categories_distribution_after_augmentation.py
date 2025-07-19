import pandas as pd

path = './train_absaPairs_aug_final.csv'

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

df = pd.read_csv("roabsa_train_aug_final.csv") 
df['all_categories'] = df['all_categories'].str.split(';')
df_exploded = df.explode('all_categories')
df_exploded['all_categories'] = df_exploded['all_categories'].str.strip()
df_exploded['data_origin_new'] = df_exploded['data_origin'].apply(lambda x: 'synthetic' if x!='manual' else x)
print(df_exploded['data_origin_new'].value_counts())


category_counts = df_exploded.groupby(['all_categories', 'data_origin_new']).size().unstack(fill_value=0)

for col in ['manual', 'synthetic']:
    if col not in category_counts.columns:
        category_counts[col] = 0

category_counts['Total'] = category_counts['manual'] + category_counts['synthetic']
category_counts = category_counts.sort_values(by='Total', ascending=False).drop(columns='Total')
print(category_counts)

fig, ax = plt.subplots(figsize=(12, 8))
bar_width = 0.35
index = range(len(category_counts))

bars_aug = ax.bar(index, category_counts['synthetic'], bar_width, label='synthetic', color="#006d36")
bars_orig = ax.bar([i + bar_width for i in index], category_counts['manual'], bar_width, label='manual', color="#a70808")

for bar in bars_aug:
    height = bar.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)

for bar in bars_orig:
    height = bar.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)

ax.set_xlabel('Categories', fontsize=16)
ax.set_ylabel('Frequencies',fontsize=20)
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(category_counts.index, rotation=90, fontsize=18)
ax.legend(fontsize=20)

plt.tight_layout()
plt.show()
