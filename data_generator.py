from dataset.CSV_Generation import *
import os


base_seed = 42

sample_num_list = {10, 100, 1000, 10000}

j = 0

for i in range (6):
    for sample_num in sample_num_list:
        unique_seed = base_seed + j * 1000 * sample_num
        if not os.path.exists(f'experiment_data/training_{sample_num}_{i}.csv'):
            generate_csv_data(sample_num, f'experiment_data/training_{sample_num}_{i}.csv', unique_seed)
        j += 1
        unique_seed = base_seed + j * 1000 * sample_num
        if not os.path.exists(f'experiment_data/validating_{sample_num}_{i}.csv'):
            generate_csv_data(sample_num, f'experiment_data/validating_{sample_num}_{i}.csv', unique_seed)
        j += 1
    unique_seed = base_seed + j * 1000
    if not os.path.exists(f'experiment_data/testing_{10000}_{i}.csv'):
        generate_csv_data(10000, f'experiment_data/testing_{10000}_{i}.csv', unique_seed)
    j += 1
