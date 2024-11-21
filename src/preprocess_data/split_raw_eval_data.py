import pandas as pd
import re, string, os, json, csv
from pathlib import Path
BASE_DIR = Path(__file__).parent

class Config:
    IN_DIR = str(BASE_DIR) + os.sep + 'raw_data'
    OUT_DIR = str(BASE_DIR) + os.sep + 'processed_data'
    FILE_NAME = 'test_eval_roabsa_new_jsonmin.json'
    FILE_NAME_EVAL_CSV = 'roabsa_eval.csv'
    FILE_NAME_TEST_CSV = 'roabsa_test.csv'
    COLS_TO_REMOVE =['lead_time','updated_at','created_at','annotator',
                    'annotation_id','sents_length','rating',
                    'tokenized_sents']



def read_file(filename):
    with open(filename, "r", encoding="utf-8-sig") as file:
        reviews_data = json.load(file)
    return reviews_data

def format_data(filename):
    json_data = read_file(filename)
    output_file = "output.csv"

    with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['review', 'aspects_with_polarities']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for json_object in json_data:
            if json_object.get("annotator") == 22:
                aspects_polarities = json_object.get("aspects_polarities")
                if 'choices' in aspects_polarities:
                    choices = aspects_polarities.get("choices")
                    aspects_polarities = ', '.join(choices)
                    writer.writerow({'review': json_object["body"], 'aspects_with_polarities': aspects_polarities})

    df = pd.read_csv(output_file)
    os.remove(output_file)

    return df

df = format_data(filename=Config.IN_DIR+os.sep+Config.FILE_NAME)
print(df)
print(df.columns)