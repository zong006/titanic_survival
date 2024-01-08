import pandas as pd
import os

def extract_data():
    current_file_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(os.path.dirname(current_file_path))
    data_directory = os.path.join(parent_directory, 'data')

    data_path = os.path.join(data_directory,'train.csv')

    df = pd.read_csv(data_path)

    return df