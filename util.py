from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import ParameterGrid


education_level = ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']
employment_status = ['Employed', 'Self-Employed', 'Unemployed']
marital_status = ['Single', 'Married', 'Divorced', 'Widowed']
home_ownership_status = ['Own', 'Rent', 'Mortgage', 'Other']
loan_purpose = ['Home', 'Auto', 'Education', 'Debt Consolidation', 'Other']


# def convert_date_to_int(date_str):
#     # Convert a date string in the form %Y-%m-%d into a int number representing the number of days since 1970-01-01
#     date_obj = datetime.strptime(date_str, "%Y-%m-%d")
#     days_since_epoch = (date_obj - datetime(1970, 1, 1)).days
#     return days_since_epoch

def convert_date_to_int(date_str):
    # Convert a date string in the form %Y-%m-%d into a int number representing the number of days since 1970-01-01
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    except ValueError:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")

    # Calculate days since epoch (1970-01-01)
    days_since_epoch = (date_obj - datetime(1970, 1, 1)).days
    return days_since_epoch



def convert_string_attr_to_boolean(target, dict_list):
    # Return a boolean list of 0 and 1 for whether target shows up at i-th bucket of the list.
    return [1 if target == item else 0 for item in dict_list]


def normalize_dataframe(df):
    # Extract features for scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Convert scaled data back to a DataFrame for usability
    processed_df = pd.DataFrame(scaled_data, columns=df.columns)

    return processed_df, scaler


def pre_process_training_dataset(csv_filename):
    df = pd.read_csv(csv_filename)
    df['ApplicationDate'] = df['ApplicationDate'].apply(convert_date_to_int)

    # Apply the conversion to the relevant columns
    for col_name, categories in zip(
            ['EducationLevel', 'EmploymentStatus', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose'],
            [education_level, employment_status, marital_status, home_ownership_status, loan_purpose]
    ):
        # Create new columns for each category
        for category in categories:
            df[f"{col_name}_{category}"] = df[col_name].apply(lambda x: int(x == category))

    # Drop the original columns after conversion
    df = df.drop(columns=['EducationLevel', 'EmploymentStatus', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose'])

    normalized_df, scaler = normalize_dataframe(df)

    return normalized_df, scaler


def pre_process_test_dataset(csv_filename, scaler):
    df = pd.read_csv(csv_filename)
    df['ApplicationDate'] = df['ApplicationDate'].apply(convert_date_to_int)

    # Apply the conversion to the relevant columns
    for col_name, categories in zip(
            ['EducationLevel', 'EmploymentStatus', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose'],
            [education_level, employment_status, marital_status, home_ownership_status, loan_purpose]
    ):
        # Create new columns for each category
        for category in categories:
            df[f"{col_name}_{category}"] = df[col_name].apply(lambda x: int(x == category))

    # Drop the original columns after conversion
    df = df.drop(columns=['EducationLevel', 'EmploymentStatus', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose'])

    # Normalize using the provided scaler
    normalized_data = scaler.transform(df)
    normalized_df = pd.DataFrame(normalized_data, columns=df.columns)

    return normalized_df


def separate_labels(df):
    label_columns = ["LoanApproved", "RiskScore"]
    y1_array = df[label_columns[0]].to_numpy()
    y2_array = df[label_columns[1]].to_numpy()
    x_df = df.drop(columns=label_columns)
    return x_df, y1_array, y2_array


def generate_dataset(x_df, y_array):
    x_tensor = torch.tensor(x_df.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_array, dtype=torch.float32).unsqueeze(1)
    return TensorDataset(x_tensor, y_tensor)


def evaluate_binary_classification(model, dataset, threshold=0.5):
    """
    Evaluate a trained binary classification model on a test or validation dataset.

    Args:
        model: The trained PyTorch model.
        dataset: A PyTorch dataset containing features and labels.
        threshold: Threshold for converting predicted probabilities to binary labels.

    Returns:
        metrics: A dictionary containing accuracy, precision, recall, F1 score,
                 and the values tp, fp, tn, fn from the confusion matrix.
    """
    # DataLoader for the dataset
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    # Ensure model is in evaluation mode
    model.eval()

    # Gather predictions and true labels
    with torch.no_grad():
        for batch_x, batch_y in loader:
            predictions = model(batch_x)
            predictions = torch.sigmoid(predictions).squeeze()
            predicted_labels = (predictions >= threshold).int()
            true_labels = batch_y.int()
    predicted_labels = predicted_labels.numpy()
    true_labels = true_labels.numpy()

    # Compute confusion matrix components
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, zero_division=0),
        "f1_score": f1_score(true_labels, predicted_labels, zero_division=0),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }

    return metrics

def find_best_hyperparameters(model_class, insize, layer_dims, train_dataset, valid_dataset, param_grid):
    """
    Find the best hyperparameters for a model using F1 Score on the validation dataset.

    param_grid example:
    param_grid = {
    'num_epochs': [20, 50, 100],
    'batch_size': [16, 32, 64],
    'learning_rate': [0.001, 0.01, 0.1]
    }
    """
    best_f1 = -1
    best_params = None

    # Iterate over all combinations of parameters
    for params in ParameterGrid(param_grid):
        # Unpack parameters
        num_epochs = params['num_epochs']
        batch_size = params['batch_size']
        learning_rate = params['learning_rate']

        # Initialize model
        model = model_class(in_size=insize, layer_dims=layer_dims)

        # Train the model
        model.train_model_binary(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        # Evaluate on the validation set
        metrics = evaluate_binary_classification(model, valid_dataset)
        f1 = metrics['f1_score']

        # Update the best parameters if the current F1 is better
        if f1 > best_f1:
            best_f1 = f1
            best_params = params

    return best_params, best_f1



def find_best_hyperparameters_regularization(model_class, insize, layer_dims, train_dataset, valid_dataset, param_grid):
    """
    Find the best hyperparameters for a model using F1 Score on the validation dataset.

    param_grid example:
    param_grid = {
    'num_epochs': [20, 50, 100],
    'batch_size': [16, 32, 64],
    'learning_rate': [0.001, 0.01, 0.1]
    'regularization_class': [1, 2]
    'regularization_lambda':[0.01, 0.001]
    }
    """
    best_f1 = -1
    best_params = None

    # Iterate over all combinations of parameters
    for params in ParameterGrid(param_grid):
        # Unpack parameters
        num_epochs = params['num_epochs']
        batch_size = params['batch_size']
        learning_rate = params['learning_rate']
        reg_class = params['regularization_class']
        reg_lambda = params['regularization_lambda']

        # Initialize model
        model = model_class(in_size=insize, layer_dims=layer_dims)

        # Train the model
        model.train_model_binary_regularization(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            reg_class=reg_class,
            reg_lambda=reg_lambda
        )

        # Evaluate on the validation set
        metrics = evaluate_binary_classification(model, valid_dataset)
        f1 = metrics['f1_score']

        # Update the best parameters if the current F1 is better
        if f1 > best_f1:
            best_f1 = f1
            best_params = params

    return best_params, best_f1


def find_best_hyperparameters_regression(model_class, insize, layer_dims, train_dataset, valid_dataset, validate_dataset_binary, param_grid):
    """
    Find the best hyperparameters and threshold for a regression model
    using ACCURACY on the validation dataset.

    Note: We note that using f1 score won't give any reasonable thresholds

    Args:
        model_class: Class of the model to be used.
        insize: Input size for the model.
        layer_dims: List of layer dimensions.
        train_dataset: Dataset for training.
        valid_dataset: Dataset for validation with risk scores (0 - 1 after standardization).
        valid_dataset: Dataset for validation with binary classes.
        param_grid:
            param_grid example:
            param_grid = {
            'num_epochs': [20, 50, 100],
            'batch_size': [16, 32, 64],
            'learning_rate': [0.001, 0.01, 0.1]
            }

    Returns:
        best_params: Best combination of hyperparameters.
        best_threshold: Threshold that gives the best F1 score.
        best_f1: Best F1 score achieved.
    """
    best_accuracy = -1
    best_params = None
    best_threshold = None

    # Iterate over all combinations of parameters
    for params in ParameterGrid(param_grid):
        # Unpack parameters
        num_epochs = params['num_epochs']
        batch_size = params['batch_size']
        learning_rate = params['learning_rate']

        # Initialize model
        model = model_class(in_size=insize, layer_dims=layer_dims)

        # Train the model
        model.train_model_regression(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        # Validation: Calculate predictions
        model.eval()
        valid_loader = DataLoader(validate_dataset_binary, batch_size=batch_size, shuffle=False)
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                outputs = model(batch_x)
                predictions.extend(outputs.squeeze().tolist())
                true_labels.extend(batch_y.tolist())

        # Find the best threshold for F1 score
        thresholds = [i * 0.01 for i in range(1, 100)]  # Test thresholds from 0.01 to 0.99
        for threshold in thresholds:
            binary_preds = [1 if pred <= threshold else 0 for pred in predictions]
            accuracy = accuracy_score(true_labels, binary_preds)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params
                best_threshold = threshold

    return best_params, best_threshold, best_accuracy


def evaluate_binary_classification_for_regression_model(model, dataset, threshold):
    """
    Evaluate a trained regression model on a test or validation dataset with binary y labels and a threshold.

    Args:
        model: The trained NN regression model (Risk scores).
        dataset: A PyTorch dataset containing features and BINARY!!! labels.
        threshold: Threshold for converting predicted probabilities to binary labels.

        Note: true --> <= threshold

    Returns:
        metrics: A dictionary containing accuracy, precision, recall, F1 score,
                 and the values tp, fp, tn, fn from the confusion matrix.
    """
    # DataLoader for the dataset
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    # Ensure model is in evaluation mode
    model.eval()

    # Gather predictions and true labels
    with torch.no_grad():
        for batch_x, batch_y in loader:
            predictions = model(batch_x)
            predicted_labels = (predictions.squeeze() <= threshold).int()  # No sigmoid here
            true_labels = batch_y.int()
    predicted_labels = predicted_labels.numpy()
    true_labels = true_labels.numpy()

    # Compute confusion matrix components
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, zero_division=0),
        "f1_score": f1_score(true_labels, predicted_labels, zero_division=0),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }

    return metrics
