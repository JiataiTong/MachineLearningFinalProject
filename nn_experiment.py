from util import *
from Network import *

param_grid = {
    'num_epochs': [20, 50, 100],
    'batch_size': [16, 32, 64],
    'learning_rate': [0.001, 0.01, 0.1]
    }

depths = [1, 2]
layer_sizes = [5, 10, 50]

dataset_size_list = [10, 100, 1000]
training_file_name = 'training_'
validation_file_name = 'validating_'
test_file_name = 'testing_'


seed = 50

for dataset_size in dataset_size_list:
    # Preprocess data
    training_df, scaler = pre_process_training_dataset(f'{training_file_name}{dataset_size}.csv')
    validation_df = pre_process_test_dataset(f'{validation_file_name}{dataset_size}.csv', scaler)
    test_df = pre_process_test_dataset(f'{test_file_name}{dataset_size}.csv', scaler)

    train_x, train_y, train_risk_score = separate_labels(training_df)
    val_x, val_y, val_risk_score = separate_labels(validation_df)
    test_x, test_y, test_risk_score = separate_labels(test_df)

    in_size = train_x.shape[1]
    # print(in_size)

    # construct datasets for binary classification
    train_dataset_binary = generate_dataset(train_x, train_y)
    val_dataset_binary = generate_dataset(val_x, val_y)
    test_dataset_binary = generate_dataset(test_x, test_y)

    for depth in depths:
        for layer_size in layer_sizes:
            layer_dims = [layer_size] * depth + [1]
            best_param, best_f1 = find_best_hyperparameters(Network, in_size, train_dataset_binary, val_dataset_binary, param_grid)
            print(f'{dataset_size}-{layer_dims}: {best_f1}')








