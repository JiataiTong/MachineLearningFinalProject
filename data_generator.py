from dataset.CSV_Generation import *
import os

sample_num_list = {10, 100, 1000, 10000}

for sample_num in sample_num_list:
    if not os.path.exists(f'training_{sample_num}.csv'):
        generate_csv_data(sample_num, f'training_{sample_num}.csv')
    if not os.path.exists(f'validating_{sample_num}.csv'):
        generate_csv_data(sample_num, f'validating_{sample_num}.csv')
    if not os.path.exists(f'testing_{sample_num}.csv'):
        generate_csv_data(sample_num, f'testing_{sample_num}.csv')
