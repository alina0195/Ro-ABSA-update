import pandas as pd

path = './train_absaPairs_aug_final.csv'
# df_absa = pd.read_csv(path)

# def extract_labels(label):
#     categories = label.split(';')
#     categories = [c.split(' is ')[0].strip() for c in categories]
#     categories = '; '.join(categories)
#     # print(categories)
#     return categories.strip()

# df_absa['atc_target'] = df_absa['absa_target'].apply(extract_labels)
# print(df_absa['atc_target'].value_counts())
# # id,text_cleaned,all_categories,data_origin

# df_atc= df_absa.copy()
# df_atc.drop(columns=['absa_target'], inplace=True)
# df_atc.rename({'absa_input':'text_cleaned','atc_target':'all_categories'}, inplace=True)
# print(df_atc.columns)
# df_atc.to_csv('./roabsa_train_aug_final.csv', index=False)

# create the distribution plot
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

df = pd.read_csv("roabsa_train_aug_final.csv")  # Replace with the actual filename if needed
df['all_categories'] = df['all_categories'].str.split(';')
df_exploded = df.explode('all_categories')
df_exploded['all_categories'] = df_exploded['all_categories'].str.strip()
df_exploded['data_origin_new'] = df_exploded['data_origin'].apply(lambda x: 'synthetic' if x!='manual' else x)
print(df_exploded['data_origin_new'].value_counts())

# category_counts = df_exploded.groupby(['data_origin_new', 'all_categories']).size().unstack(fill_value=0)
# category_counts['total'] = category_counts.sum(axis=1)
# category_counts = category_counts.sort_values('total', ascending=False).drop(columns='total')

# Reindex the categories based on descending order of Augmented values
# category_counts = category_counts.loc[:, category_counts.loc['synthetic'].sort_values(ascending=False).index]



# category_counts.T.plot(kind='bar', figsize=(10, 6))
# plt.xlabel('Categories')
# plt.ylabel('Frequencies')
# plt.title('Category Frequencies in Original vs Synthetic Data')
# plt.xticks(rotation=90)
# plt.legend(title='Data Origin')
# plt.tight_layout()
# plt.show()


# Count frequencies by data_origin and category
category_counts = df_exploded.groupby(['all_categories', 'data_origin_new']).size().unstack(fill_value=0)

# Ensure both Original and Augmented columns are present
for col in ['manual', 'synthetic']:
    if col not in category_counts.columns:
        category_counts[col] = 0

# Sort by total frequency
category_counts['Total'] = category_counts['manual'] + category_counts['synthetic']
category_counts = category_counts.sort_values(by='Total', ascending=False).drop(columns='Total')
print(category_counts)

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
bar_width = 0.35
index = range(len(category_counts))

bars_aug = ax.bar(index, category_counts['synthetic'], bar_width, label='synthetic', color="#006d36")
bars_orig = ax.bar([i + bar_width for i in index], category_counts['manual'], bar_width, label='manual', color="#a70808")

# Add text labels
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

# Customize plot
ax.set_xlabel('Categories', fontsize=16)
ax.set_ylabel('Frequencies',fontsize=20)
# ax.set_title('Frequencies by Category and Data Origin')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(category_counts.index, rotation=90, fontsize=18)
ax.legend(fontsize=20)

plt.tight_layout()
plt.show()


# category_counts = df_exploded['all_categories'].value_counts()
# category_counts = category_counts.sort_values(ascending=False)

# # Plotting
# fig, ax = plt.subplots(figsize=(12, 6))
# bars = ax.bar(category_counts.index, category_counts.values, color='brown')

# # Add number labels above bars
# for bar in bars:
#     height = bar.get_height()
#     ax.annotate(f'{int(height)}',
#                 xy=(bar.get_x() + bar.get_width() / 2, height),
#                 xytext=(0, 3),
#                 textcoords="offset points",
#                 ha='center', va='bottom')

# # Customize plot
# ax.set_xlabel('Categories')
# ax.set_ylabel('Frequencies')
# ax.set_title('Frequencies by Category (All Polarities Combined)')
# ax.set_xticklabels(category_counts.index, rotation=90)

# plt.tight_layout()
# plt.show()


# # fig, ax = plt.subplots(figsize=(10, 6))
# # bar_width = 0.35
# # index = range(len(category_counts))

# # bars_aug = ax.bar(index, category_counts['synthetic'], bar_width, label='synthetic', color="#188a09")
# # bars_orig = ax.bar([i + bar_width for i in index], category_counts['Original'], bar_width, label='Original', color='#ff7f0e')

# # # Adding frequency labels above the bars
# # for bar in bars_aug:
# #     height = bar.get_height()
# #     ax.annotate(f'{int(height)}',
# #                 xy=(bar.get_x() + bar.get_width() / 2, height),
# #                 xytext=(0, 3),
# #                 textcoords="offset points",
# #                 ha='center', va='bottom')

# # for bar in bars_orig:
# #     height = bar.get_height()
# #     ax.annotate(f'{int(height)}',
# #                 xy=(bar.get_x() + bar.get_width() / 2, height),
# #                 xytext=(0, 3),
# #                 textcoords="offset points",
# #                 ha='center', va='bottom')

# # # Customize axes and layout
# # ax.set_xlabel('Categories')
# # ax.set_ylabel('Frequencies')
# # ax.set_title('Frequencies by Category and Data Origin')
# # ax.set_xticks([i + bar_width / 2 for i in index])
# # ax.set_xticklabels(category_counts.index, rotation=90)
# # ax.legend()

# # plt.tight_layout()
# # plt.show()