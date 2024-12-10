from util import *
from Network import *
import os
from datetime import datetime
from noise_utils import add_feature_noise, add_label_noise

current_time = datetime.now().strftime("%y%m%d%H%M")
res_filename = f'result/result_nn_{current_time}.csv'

sample_results_df = pd.DataFrame([{
    "dataset_size": 1000,
    "layer_dims": "[64, 32, 1]",
    "accuracy": 0.95,
    "precision": 0.92,
    "recall": 0.93,
    "f1": 0.925,
    "noise_level": 0.01,
    "model_type": "NN"
}])

sample_results_df.head(0).to_csv(res_filename, index=False)
print(f"CSV file '{res_filename}' created successfully.")





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

noise_levels = [0.0, 0.01, 0.05, 0.1]

for dataset_size in dataset_size_list:
    training_df, scaler = pre_process_training_dataset(f'{training_file_name}{dataset_size}.csv')
    validation_df = pre_process_test_dataset(f'{validation_file_name}{dataset_size}.csv', scaler)
    test_df = pre_process_test_dataset(f"testing_10000.csv", scaler)

    train_x, train_y, train_risk_score = separate_labels(training_df)
    val_x, val_y, val_risk_score = separate_labels(validation_df)
    test_x, test_y, test_risk_score = separate_labels(test_df)

    in_size = train_x.shape[1]

    for noise_level in noise_levels:
        noisy_train_x = add_feature_noise(train_x, noise_std=noise_level)
        noisy_train_y = train_y.copy()

        train_dataset_binary = generate_dataset(noisy_train_x, noisy_train_y)
        val_dataset_binary = generate_dataset(val_x, val_y)
        test_dataset_binary = generate_dataset(test_x, test_y)

        for depth in depths:
            for layer_size in layer_sizes:
                layer_dims = [layer_size] * depth + [1]

                best_param, best_f1 = find_best_hyperparameters(
                    model_class=Network, 
                    insize=in_size, 
                    layer_dims=layer_dims,
                    train_dataset=train_dataset_binary, 
                    valid_dataset=val_dataset_binary, 
                    param_grid=param_grid
                )

                model = Network(in_size=in_size, layer_dims=layer_dims)
                model.train_model_binary(
                    train_dataset=train_dataset_binary,
                    valid_dataset=val_dataset_binary,
                    num_epochs=best_param["num_epochs"],
                    batch_size=best_param["batch_size"],
                    learning_rate=best_param["learning_rate"]
                )

                metrics = evaluate_binary_classification(model, test_dataset_binary)

                results_df = pd.DataFrame([{
                    "dataset_size": dataset_size,
                    "layer_dims": str(layer_dims),
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1_score"],
                    "noise_level": noise_level,
                    "model_type": "NN"
                }])

                results_df.to_csv(res_filename, mode='a', header=not os.path.exists(res_filename), index=False)
                print(f"Results for dataset {dataset_size}, depth {depth}, layer_size {layer_size}, noise_level {noise_level} appended.")