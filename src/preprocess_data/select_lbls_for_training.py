import pandas as pd
import re, os
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent

class Config:
    SEED=42
    IN_DIR = str(BASE_DIR) + os.sep + 'raw_data'
    OUT_DIR = str(BASE_DIR) + os.sep + 'processed_data'
        
    FILE_NAME_TRAIN_MANUAL = os.sep + 'old_train' + os.sep +'df_train_manual.csv'
    # FILE_NAME_TRAIN_MANUAL = os.sep + 'old_train' + os.sep +'df_train_augmented_almostRare.csv'
    # FILE_NAME_TRAIN_MANUAL = os.sep + 'old_train' + os.sep +'df_train_augmented_rare.csv'
    
    FILE_NAME = 'train_roabsa_new_jsonmin.json'
    FILE_NAME_INCONSISTENCES = 'roabsa_train_inconsistences.csv'
    FILE_NAME_SAVE = 'roabsa_train_lbl_new.csv'
    FILE_NAME_SAVE_PROCESSED = 'roabsa_train_lbl_new_process.csv'
    
    
df = pd.read_csv(Config.OUT_DIR + os.sep + Config.FILE_NAME_SAVE_PROCESSED)

