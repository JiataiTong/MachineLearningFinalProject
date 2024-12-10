import numpy as np

def add_feature_noise(x_df, noise_std=0.01):
    """
    Add Gaussian noise to the features.
    :param x_df: pandas DataFrame of features
    :param noise_std: Standard deviation of Gaussian noise to add
    :return: pandas DataFrame of noisy features
    """
    noisy_x = x_df.copy()
    noise = np.random.normal(loc=0.0, scale=noise_std, size=noisy_x.shape)
    noisy_x += noise
    return noisy_x


def add_label_noise(y_array, flip_ratio=0.01):
    """
    Flip a percentage of labels for label noise.
    Assumes binary labels {0,1}.
    :param y_array: numpy array of labels
    :param flip_ratio: fraction of labels to flip
    :return: numpy array of noisy labels
    """
    y_noisy = y_array.copy()
    num_flips = int(len(y_noisy) * flip_ratio)
    flip_indices = np.random.choice(len(y_noisy), num_flips, replace=False)
    y_noisy[flip_indices] = 1 - y_noisy[flip_indices]
    return y_noisy