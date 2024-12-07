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


def convert_date_to_int(date_str):
    # Convert a date string in the form %Y-%m-%d into a int number representing the number of days since 1970-01-01
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
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
            predictions = model(batch_x)  # Forward pass
            predictions = torch.sigmoid(predictions).squeeze()  # Apply sigmoid and flatten
            predicted_labels = (predictions >= threshold).int()  # Convert probabilities to binary labels
            true_labels = batch_y.int()  # Ensure labels are integers for comparison

    # Convert tensors to NumPy arrays for evaluation
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
    best_f1 = 0
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



