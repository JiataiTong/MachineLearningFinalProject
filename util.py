from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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




