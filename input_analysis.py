
from util import *
from Network import *
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(60)

layer_dims = [5, 5, 1]
training_file_name = 'training_'
validation_file_name = 'validating_'
test_file_name = 'testing_'

dataset_size = 10000

training_df, scaler = pre_process_training_dataset(f"experiment_data/training_{dataset_size}.csv")
validation_df = pre_process_test_dataset(f"experiment_data/validating_{dataset_size}.csv", scaler)

train_x, train_y, train_risk_score = separate_labels(training_df)
val_x, val_y, val_risk_score = separate_labels(validation_df)

in_size = train_x.shape[1]

train_dataset_binary = generate_dataset(train_x, train_y)
val_dataset_binary = generate_dataset(val_x, val_y)


# print(train_x)

random_sample = val_x.sample(n=1).iloc[0]

# Build a dataset for total assets
total_assets_values = np.linspace(0, 1, 1001)
df_total_assets = pd.DataFrame([random_sample] * len(total_assets_values))
df_total_assets["TotalAssets"] = total_assets_values

# Build a dataset for total
job_tenure_values = np.linspace(0, 1, 1001)
df_job_tenure = pd.DataFrame([random_sample] * len(job_tenure_values))
df_job_tenure["JobTenure"] = job_tenure_values


# From our experiment data
best_param = {'batch_size': 64, 'learning_rate': 0.001, 'num_epochs': 200}


# Initialize and train the model
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

# Now forward our generated inputs to the model
df_total_assets_tensor = torch.tensor(df_total_assets.values, dtype=torch.float32)
df_job_tenure_tensor = torch.tensor(df_job_tenure.values, dtype=torch.float32)

total_assets_outputs = model(df_total_assets_tensor).detach().numpy()
total_assets_values = df_total_assets["TotalAssets"].values

job_tenure_outputs = model(df_job_tenure_tensor).detach().numpy()
job_tenure_values = df_job_tenure["JobTenure"].values

# Plot TotalAssets vs. output values
plt.figure()
plt.plot(total_assets_values, total_assets_outputs, label="TotalAssets vs Output")
plt.xlabel("TotalAssets/JobTenure")
plt.ylabel("Model Output")
plt.title("TotalAssets/JobTenure vs. Model Output")
# plt.legend()
# plt.grid()

# Plot JobTenure vs. output values
# plt.figure()
plt.plot(job_tenure_values, job_tenure_outputs, label="JobTenure vsOutput", color="orange")
# plt.xlabel("JobTenure")
# plt.ylabel("Model Output")
# plt.title("JobTenure vs. Model Output")
plt.legend()
plt.grid()

plt.show()