import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.random import randint

def load_csv_data(folder_path, label_file, behavioral_features, csvs):
    """
    Load data from CSV files in a folder and corresponding labels.

    Args:
        folder_path (str): Path to the folder containing the CSV files.
        label_file (str): Path to the CSV file containing the labels.
        behavioral_features (list of str): List of behavioral features to extract from the data.

    Returns:
        np.array: Array of data extracted from the CSV files.
        np.array: Array of labels corresponding to the data.
    """
        
    labels_df = pd.read_csv(label_file)
    labels_df['chunk'] = labels_df['chunk'].astype(str)
    #EmotiW
    labels_df['label'] = labels_df['label'].apply(process_label)
    # labels_df['Engagement'] =labels_df['Engagement'].astype(int) 
    all_data, all_labels = [], []

    # Process each CSV file in the directory
    for filename in tqdm(os.listdir(folder_path), desc="Loading data"):
        if filename.endswith('.csv') and filename in csvs:
            subject_id = filename.split('.')[0]
            subject_file = os.path.join(folder_path, filename)
            subject_data = pd.read_csv(subject_file)
            

            # subject_data['EAR'] = ((abs(subject_data['y_37'] - subject_data['y_41']) + abs(subject_data['y_38'] - subject_data['y_40'])) / (2 * abs(subject_data['x_36'] - subject_data['x_39'])) 
            #                        + (abs(subject_data['y_43'] - subject_data['y_47']) + abs(subject_data['y_44'] - subject_data['y_46']) ) / (2 * abs(subject_data['x_42'] - subject_data['x_45']))
            #                        ) / 2
            # subject_data['EAR'] = subject_data['EAR'].replace([np.inf, -np.inf], 1e-3)

            # subject_data['EAR'].apply(lambda x: x if x == else 1e-3)

            # subject_data['EAR'] = 2 * (subject_data['EAR'] - subject_data['EAR'].min()) / (subject_data['EAR'].max() - subject_data['EAR'].min()) - 1

#         df = df[['EAR','pose_Tx', 'pose_Ty']] 

            # Stack data for selected behavioral features
            subject_data_values = np.stack([subject_data[col].values for col in behavioral_features], axis=0)
            subject_label = labels_df[labels_df['chunk'].str.contains(subject_id)]['label'].values
            _, k = subject_data_values.shape
            num_segments = 16
            average_duration = k //16 #划分为16段
            # if average_duration > 0:
            #     offsets = np.multiply(list(range(num_segments)), average_duration) + randint(average_duration, size=num_segments)
            # elif k > num_segments:
            #     offsets = np.sort(randint(k, size=num_segments))
            # else:
            #     offsets = np.pad(np.arange(k), (0, num_segments - k), 'edge')

            # subject_data_values = subject_data_values[:, offsets] 
            if k > 280:
                subject_data_values = subject_data_values[:, :280]
            elif k < 280:
                padding = 280 - k
                subject_data_values = np.concatenate([subject_data_values, np.tile(subject_data_values[:, -1:], (1, padding))], axis=1)
            # Append data and label if label exists
            if len(subject_label) > 0:
                all_data.append(subject_data_values)
                all_labels.append(subject_label[0])
            else:
                print(f"No label found for subject {subject_id}")
            # break
    # Reshape the collected data and convert it to numpy arrays
    all_data = np.array(all_data)
    all_data = np.expand_dims(all_data, axis=1)  
    all_labels = np.array(all_labels)

    return all_data, all_labels

def process_label(x):

    if x == 'Highly-Engaged':
        return 3
    elif x == 'Engaged':
        return 2
    elif x == 'Barely-engaged':
        return 1
    elif x == 'Not-Engaged':
        return 0

    print(x)
    print('xxxxx')
    return 1 / 0

def get_source_data(train_folder_path, test_folder_path, valid_folder_path, label_file, behavioral_features, train_csvs, test_csvs, valid_csvs):
    """
    Load and preprocess training and testing data from the specified folders.

    Args:
        train_folder_path (str): Path to the folder containing the training data.
        test_folder_path (str): Path to the folder containing the testing data.
        label_file (str): Path to the CSV file containing the labels.
        behavioral_features (list of str): List of behavioral features to extract from the data.

    Returns:
        np.array: Processed training data.
        np.array: Labels for the training data.
        np.array: Processed testing data.
        np.array: Labels for the testing data.
    """

    # Load training data
    print('\nLoading train data ...')
    train_data, train_labels = load_csv_data(train_folder_path, label_file, behavioral_features, train_csvs)
    train_labels = train_labels.reshape(1, -1)

    # Shuffle the training data
    shuffle_index = np.random.permutation(len(train_data))
    train_data = train_data[shuffle_index, :, :, :]
    train_labels = train_labels[0][shuffle_index]
    # Load test data
    print('\nLoading test data ...')
    test_data, test_labels = load_csv_data(test_folder_path, label_file, behavioral_features, test_csvs)
    test_labels = test_labels.reshape(-1)  


    # Load validation data
    print('\nLoading validation data ...')
    valid_data, valid_labels = load_csv_data(valid_folder_path, label_file, behavioral_features, valid_csvs)
    valid_labels = valid_labels.reshape(-1)  

    # Standardize both train and test data using training data statistics
    target_mean = np.mean(train_data)
    target_std = np.std(train_data)
    train_data = (train_data - target_mean) / target_std
    test_data = (test_data - target_mean) / target_std
    valid_data = (valid_data - target_mean) / target_std
    

    return train_data, train_labels, test_data, test_labels, valid_data, valid_labels


def get_source_data_inference(inference_folder_path, label_file_inference,
                              behavioral_features, target_mean, target_std):
    """
    Load and preprocess inference data from the specified folder.

    Args:
        inference_folder_path (str): Path to the folder containing the inference data.
        label_file_inference (str): Path to the CSV file containing the labels.
        behavioral_features (list of str): List of behavioral features to extract from the data.
        target_mean (float): Mean value of the training data used for standardization.
        target_std (float): Standard deviation of the training data used for standardization.

    Returns:
        np.array: Processed testing data.
        np.array: Labels for the testing data.
    """

    # Load inference data
    print('\nLoading data for inference ...')
    inference_data, inference_labels = load_csv_data(inference_folder_path, label_file_inference, behavioral_features)
    inference_labels = inference_labels.reshape(-1)

    # Standardize inference data using provided training data statistics
    inference_data = (inference_data - target_mean) / target_std

    return inference_data, inference_labels
