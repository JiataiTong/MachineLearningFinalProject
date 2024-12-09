from util import *
from Network import *
import os
from datetime import datetime


current_time = datetime.now().strftime("%y%m%d%H%M")
res_filename = f'result/result_nn_large_{current_time}.csv'

sample_results_df = pd.DataFrame([{
    "dataset_size": 1000,
    "layer_dims": "[64, 32, 1]",
    "hyperparameters": " ",
    "accuracy": 0.95,
    "precision": 0.92,
    "recall": 0.93,
    "f1": 0.925,
    "tp": 0,
    "fp": 0,
    "tn": 0,
    "fn": 0
}])

sample_results_df.head(0).to_csv(res_filename, index=False)
print(f"CSV file '{res_filename}' created successfully.")





param_grid = {
    'num_epochs': [20, 50, 100, 200],
    'batch_size': [32, 64],
    'learning_rate': [0.001, 0.01, 0.1]
    }

depths = [1, 2]
layer_sizes = [5, 10, 50]

dataset_size_list = [10000, 100000]
training_file_name = 'training_'
validation_file_name = 'validating_'
test_file_name = 'testing_'


seed = 50

test_data_size = 10000


for dataset_size in dataset_size_list:
    # Preprocess data
    training_df, scaler = pre_process_training_dataset(f'experiment_data/{training_file_name}{dataset_size}.csv')
    validation_df = pre_process_test_dataset(f'experiment_data/{validation_file_name}{dataset_size}.csv', scaler)
    test_df = pre_process_test_dataset(f'experiment_data/{test_file_name}{test_data_size}.csv', scaler)

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
            best_param, best_f1 = find_best_hyperparameters(Network, in_size, layer_dims, train_dataset_binary, val_dataset_binary, param_grid)
            # print(f'{dataset_size}-{layer_dims}: {best_f1}')

            # Apply the best parameter and get the result
            model = Network(
                in_size=in_size,
                layer_dims=layer_dims
            )

            model.train_model_binary(
                train_dataset=train_dataset_binary,
                valid_dataset=val_dataset_binary,
                num_epochs=best_param["num_epochs"],
                batch_size=best_param["batch_size"],
                learning_rate=best_param["learning_rate"]
            )

            # Evaluate the model on the test dataset
            metrics = evaluate_binary_classification(model, test_dataset_binary)

            # Prepare the results DataFrame
            results_df = pd.DataFrame([{
                "dataset_size": dataset_size,
                "layer_dims": str(layer_dims),
                "hyperparameters": best_param,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1_score"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "tn": metrics["tn"],
                "fn": metrics["fn"]
            }])

            results_df.to_csv(res_filename, mode='a', header=not os.path.exists(res_filename), index=False)
            print(f"Results successfully appended to {res_filename}.")







