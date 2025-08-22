#-----------------data.py-----------------
import yaml
import numpy as np
import pandas as pd


def preprocess_file(file_path, sequence_length=1000):
    """Preprocess a single gait file and return segments"""
    try:
        # Load data (assuming tab-separated values)
        data = pd.read_csv(file_path, sep='\t', header=None)

        if data.shape[1] > 1:
            features = data.iloc[:, 1:-2].values  # Skip time column
        else:
            features = data.values
        
        # Create segments of desired length
        segments = []
        if len(features) >= sequence_length:
            # Create overlapping segments
            step_size = sequence_length  # 50% overlap
            for i in range(0, len(features) - sequence_length + 1, step_size):
                segment = features[i:i+sequence_length]
                
                # Reshape for CNN input (channels, length)
                segment = segment.T  # Shape: (n_features, sequence_length)
                segments.append(segment)
        else:
            # Pad if too short
            padding = np.zeros((sequence_length - len(features), features.shape[1]))
            features = np.vstack([features, padding])

            
            # Reshape for CNN input (channels, length)
            features = features.T  # Shape: (n_features, sequence_length)
            segments.append(features)
        
        return segments
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    

# Load patients data from YAML
def load_data():
    with open("data.yaml") as f:
        return yaml.safe_load(f)




