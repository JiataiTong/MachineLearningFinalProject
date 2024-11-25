from dataset.CSV_Generation import *

sample_num_list = {10, 100, 1000}

for sample_num in sample_num_list:
    generate_csv_data(sample_num, f'training_{sample_num}.csv')
    generate_csv_data(sample_num, f'validating_{sample_num}.csv')
    generate_csv_data(sample_num, f'testing_{sample_num}.csv')
