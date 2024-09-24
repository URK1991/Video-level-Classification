import os
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from sklearn.utils.class_weight import compute_class_weight

def load_video_features(src_dir, pad_len=188):
    """
    Load video features from Excel files, pad them to the specified length, and return the features and filenames.
    
    Parameters:
    - src_dir: directory containing Excel files with video features.
    - pad_len: maximum number of frames (features will be padded to this length).
    
    Returns:
    - feature_layers: list of padded video features.
    - vid_files: list of video filenames.
    """
    feature_layers = []
    vid_files = []
    
    for fname in os.listdir(src_dir):
        xlsx = pd.read_excel(os.path.join(src_dir, fname), index_col=0)
        vid_file = fname.split('.')[0]
        vid_features = [list(row) for _, row in xlsx.iterrows()]
        
        # Pad the features to the desired length
        n_features = len(vid_features[0])
        while len(vid_features) < pad_len:
            vid_features.append([0] * n_features)

        feature_layers.append([vid_features])
        vid_files.append(vid_file)
        print(f"Processed: {vid_file}")

    return np.asarray(feature_layers), vid_files

def create_dataframe(vid_files, feature_layers):
    """
    Create a DataFrame with video filenames and their corresponding feature layers.
    
    Parameters:
    - vid_files: list of video filenames.
    - feature_layers: list of feature layers corresponding to each video.
    
    Returns:
    - DataFrame containing 'videos' and 'feature_layer' columns.
    """
    return pd.DataFrame({'videos': vid_files, 'feature_layer': list(feature_layers)})

def merge_with_labels(features_df, labels_path):
    """
    Merge the feature DataFrame with labels from an Excel file.
    
    Parameters:
    - features_df: DataFrame containing the video features.
    - labels_path: path to the Excel file containing labels.
    
    Returns:
    - Merged DataFrame with video features and labels.
    """
    labels = pd.read_excel(labels_path)
    merged_df = features_df.merge(labels, how='right', on='videos')
    return merged_df

def save_dataframe_to_pickle(df, output_path):
    """
    Save the DataFrame to a pickle file.
    
    Parameters:
    - df: DataFrame to be saved.
    - output_path: destination file path for the pickle file.
    """
    df.to_pickle(output_path)
    print(f"Data saved to {output_path}")

def main():
    # Directory containing video feature Excel files
    src_dir = ' '
    labels_path = '.xlsx'
    output_pickle_path = '.pkl'
    
    # Load video features
    pad_len = 188 #maximum number of frames in a video within the dataset
  
    feature_layers, vid_files = load_video_features(src_dir, pad_len)
    
    # Create a DataFrame for features and video names
    features_df = create_dataframe(vid_files, feature_layers)
    
    # Merge the features with labels
    merged_df = merge_with_labels(features_df, labels_path)
    
    # Save the final DataFrame to a pickle file
    save_dataframe_to_pickle(merged_df, output_pickle_path)

if __name__ == "__main__":
    main()
