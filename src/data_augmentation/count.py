import os
import pandas as pd

ABSA_DIR = './absa_augmentation'
ATC_DIR = './atc_augmentation'


def read_from_dir(dir_name):
    csv_files = []
    for root, dirs, files in os.walk(dir_name):
        print(root,'\n', dirs,'\n', files)
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                df = pd.read_csv(full_path)
                print(df['data_origin'].value_counts())
                csv_files.append(df)
    return csv_files

absa_files = read_from_dir(ABSA_DIR)