import pandas as pd
import matplotlib.pyplot as plt

res_filename = "result/result_logistic_2412100043.csv"  # change with correct file name

df = pd.read_csv(res_filename)

models = df["model_type"].unique()

for model_type in models:
    subset = df[df["model_type"] == model_type]
    grouped = subset.groupby("noise_level")["f1"].mean().reset_index()
    plt.plot(grouped["noise_level"], grouped["f1"], marker='o', label=model_type)

plt.xlabel("Noise Level")
plt.ylabel("F1 Score")
plt.title("Noise Robustness in LR")
plt.legend()
plt.show()