import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import util

filename = '../training_10.csv'

df, scaler = util.pre_process_training_dataset(filename)

print(f'scaler: {scaler}')
df.to_csv('pre_process_training_test.csv', index=False)
