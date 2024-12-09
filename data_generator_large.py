from dataset.CSV_Generation import *
import os

sample_num = 10000

seed1 = 101
seed2 = 102
seed3 = 103

if not os.path.exists(f'experiment_data/training_{sample_num}.csv'):
    generate_csv_data(sample_num, f'experiment_data/training_{sample_num}.csv', seed1)
if not os.path.exists(f'experiment_data/validating_{sample_num}.csv'):
    generate_csv_data(sample_num, f'experiment_data/validating_{sample_num}.csv', seed2)
if not os.path.exists(f'experiment_data/testing_{sample_num}.csv'):
    generate_csv_data(sample_num, f'experiment_data/testing_{sample_num}.csv', seed3)
