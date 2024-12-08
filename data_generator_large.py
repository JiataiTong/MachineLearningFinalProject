from dataset.CSV_Generation import *
import os

sample_num = 100000


if not os.path.exists(f'experiment_data/training_{sample_num}.csv'):
    generate_csv_data(sample_num, f'experiment_data/training_{sample_num}.csv')
if not os.path.exists(f'experiment_data/validating_{sample_num}.csv'):
    generate_csv_data(sample_num, f'experiment_data/validating_{sample_num}.csv')
if not os.path.exists(f'experiment_data/testing_{sample_num}.csv'):
    generate_csv_data(sample_num, f'experiment_data/testing_{sample_num}.csv')
