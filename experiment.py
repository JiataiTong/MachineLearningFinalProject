from util import *
from Network import *

param_grid = {
    'num_epochs': [20, 50, 100],
    'batch_size': [16, 32, 64],
    'learning_rate': [0.001, 0.01, 0.1]
    }

dataset_size_list = [10, 100, 1000, 10000]
training_file_name = 'training_'
validation_file_name = 'validating_'
test_file_name = 'testing_'




for dataset_size in dataset_size_list:
    # Read Training data
    training_df, scaler = pre_process_training_dataset(f'{training_file_name}{dataset_size}.csv')
    validation_df = pre_process_test_dataset(f'{validation_file_name}{dataset_size}.csv', scaler)
    test_df = pre_process_test_dataset(f'{test_file_name}{dataset_size}.csv', scaler)

    train_x, train_y, train_risk_score = separate_labels(training_df)
    val_x, val_y, val_risk_score = separate_labels(validation_df)
    test_x, test_y, test_risk_score = separate_labels(test_df)

    train_dataset_binary = generate_dataset(train_x, train_y)
    val_dataset_binary = generate_dataset(val_x, val_y)
    test_dataset_binary = generate_dataset(test_x, test_y)

    




