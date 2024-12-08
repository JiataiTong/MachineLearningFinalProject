from dataset.CSV_Generation import *
import os

sample_num_list = {10, 100, 1000, 10000}

for i in range (6):
    for sample_num in sample_num_list:
        if not os.path.exists(f'experiment_data/training_{sample_num}_{i}.csv'):
            generate_csv_data(sample_num, f'experiment_data/training_{sample_num}_{i}.csv')
        if not os.path.exists(f'experiment_data/validating_{sample_num}_{i}.csv'):
            generate_csv_data(sample_num, f'experiment_data/validating_{sample_num}_{i}.csv')
    if not os.path.exists(f'experiment_data/testing_{10000}_{i}.csv'):
        generate_csv_data(10000, f'experiment_data/testing_{10000}_{i}.csv')
