from util import *
from logistic_regression import LogisticRegression
from datetime import datetime
from noise_utils import add_feature_noise, add_label_noise
from noise_utils import add_feature_noise, add_label_noise
import os

# Initialize result file
current_time = datetime.now().strftime("%y%m%d%H%M")
res_filename = f"result/result_logistic_{current_time}.csv"
pd.DataFrame(
    columns=[
        "dataset_size",
        "num_epochs",
        "batch_size",
        "learning_rate",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "noise_level",
        "model_type",
    ]
).to_csv(res_filename, index=False)

param_grid = {
    "num_epochs": [20, 50, 100],
    "batch_size": [16, 32, 64],
    "learning_rate": [0.001, 0.01, 0.1],
}
dataset_size_list = [10, 100, 1000]


for dataset_size in dataset_size_list:
    train_df, scaler = pre_process_training_dataset(f"training_{dataset_size}.csv")
    val_df = pre_process_test_dataset(f"validating_{dataset_size}.csv", scaler)
    test_df = pre_process_test_dataset("testing_10000.csv", scaler)
    train_x, train_y, _ = separate_labels(train_df)
    val_x, val_y, _ = separate_labels(val_df)
    test_x, test_y, _ = separate_labels(test_df)

    noise_levels = [0.0, 0.01, 0.05, 0.1]

    for noise_level in noise_levels:
        noisy_train_x = add_feature_noise(train_x, noise_std=noise_level)
        noisy_train_y = train_y.copy()

        noisy_train_dataset = generate_dataset(noisy_train_x, noisy_train_y)
        val_dataset = generate_dataset(val_x, val_y)
        test_dataset = generate_dataset(test_x, test_y)
        in_size = train_x.shape[1]

    

        best_f1, best_params = 0, {}

        for num_epochs in param_grid["num_epochs"]:
            for batch_size in param_grid["batch_size"]:
                for learning_rate in param_grid["learning_rate"]:
                    model = LogisticRegression(input_size=in_size)
                    model.train_model(
                        noisy_train_dataset, val_dataset, num_epochs, batch_size, learning_rate
                    )
                    f1_score = evaluate_binary_classification(model, val_dataset)[
                        "f1_score"
                    ]

                    if f1_score > best_f1:
                        best_f1 = f1_score
                        best_params = {
                            "num_epochs": num_epochs,
                            "batch_size": batch_size,
                            "learning_rate": learning_rate,
                        }


        model = LogisticRegression(input_size=in_size)
        model.train_model(noisy_train_dataset, val_dataset, **best_params)
        metrics = evaluate_binary_classification(model, test_dataset)

        result_dict = {
            "dataset_size": dataset_size,
            "num_epochs": best_params.get("num_epochs", None),
            "batch_size": best_params.get("batch_size", None),
            "learning_rate": best_params.get("learning_rate", None),
            "accuracy": metrics.get("accuracy", None),
            "precision": metrics.get("precision", None),
            "recall": metrics.get("recall", None),
            "f1": metrics.get("f1_score", None),
            "noise_level": noise_level,
            "model_type": "LR",  
        }
        pd.DataFrame([result_dict]).to_csv(
            res_filename, mode="a", header=False, index=False
        )

        print(f"Results for dataset size {dataset_size}, noise_level {noise_level} saved.")