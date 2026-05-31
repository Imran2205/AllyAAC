import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd
import os
import glob
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from scipy import signal
import matplotlib.pyplot as plt
from natsort import natsorted
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


###########################################
# 1. DATA LOADING AND PREPROCESSING
###########################################

def load_imu_data_from_csv(csv_path):
    """
    Load IMU data from a CSV file

    Parameters:
    - csv_path: Path to CSV file

    Returns:
    - DataFrame with IMU data
    """
    try:
        data = pd.read_csv(csv_path)
        required_columns = ['Timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

        for col in required_columns:
            if col not in data.columns:
                print(f"Warning: Column {col} not found in {csv_path}")

        return data
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def preprocess_imu_data(data, config):
    """
    Preprocess raw IMU data for gesture recognition with transformers

    Parameters:
    - data: DataFrame with columns for acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, etc.
    - config: Dictionary containing configuration parameters

    Returns:
    - windowed_data: DataFrame with features for each window
    - window_indices: Start and end indices for each window
    - scaler: The scaler used for normalization
    """
    window_size = config.get('window_size', 128)
    overlap = config.get('overlap', 0.75)
    sampling_rate = config.get('sampling_rate', 50)

    b, a = signal.butter(3, 20 / (sampling_rate / 2), 'low')
    for col in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
        if col in data.columns:
            data[col] = signal.filtfilt(b, a, data[col])

    stride = int(window_size * (1 - overlap))
    windows = []
    window_indices = []

    for start_idx in range(0, len(data), stride):
        end_idx = min(start_idx + window_size, len(data))

        if end_idx - start_idx < window_size * 0.33: 
            continue

        window = data.iloc[start_idx:end_idx].copy()
        if end_idx - start_idx < window_size:
            x = np.linspace(0, 1, end_idx - start_idx)
            x_new = np.linspace(0, 1, window_size)
            window_array = window.values
            resized_window = np.zeros((window_size, window_array.shape[1]))
            for i in range(window_array.shape[1]):
                resized_window[:, i] = np.interp(x_new, x, window_array[:, i])
            window = pd.DataFrame(resized_window, columns=window.columns)
        windows.append(window)
        window_indices.append((start_idx, end_idx))

    if len(windows) == 0:
        return None, None, None

    windowed_data = []
    imu_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

    # Add magnetometer columns if they exist
    # if all(col in data.columns for col in ['magn_x', 'magn_y', 'magn_z']):
    #     imu_columns.extend(['magn_x', 'magn_y', 'magn_z'])

    for window in windows:
        window_data = window[imu_columns].values
        windowed_data.append(window_data)

    windowed_data = np.array(windowed_data)

    original_shape = windowed_data.shape
    windowed_data_2d = windowed_data.reshape(-1, windowed_data.shape[-1])

    if 'normalization_stat' in config and config['normalization_stat'] is not None:
        norm_stats = config['normalization_stat']
        normalized_data_2d = (windowed_data_2d - norm_stats['mean']) / norm_stats['std']
        scaler = None  
    else:
        print("WARNING: Cannot find normalization stat.")
        scaler = StandardScaler()
        normalized_data_2d = scaler.fit_transform(windowed_data_2d)
    normalized_data = normalized_data_2d.reshape(original_shape)

    return normalized_data, window_indices, scaler


###########################################
# 2. MODEL ARCHITECTURE
###########################################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=1000):
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class IMUTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, num_classes=0,
                 dropout=0.1, max_seq_length=128, task='pretraining',
                 temporal_kernel_size=8, temporal_stride=1):
        super().__init__()

        self.task = task
        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_stride = temporal_stride
        self.input_dim = input_dim

        # Encoder to convert raw IMU data into token embeddings
        self.temporal_tokenizer = nn.Conv1d(
            in_channels=input_dim,
            out_channels=d_model,
            kernel_size=temporal_kernel_size,
            stride=temporal_stride,
            padding=0
        )

        adjusted_max_length = (max_seq_length - temporal_kernel_size) // temporal_stride + 1
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        if task == 'classification':
            self.classifier = nn.Linear(d_model, num_classes)
        elif task == 'pretraining':
            self.temporal_decoder = nn.ConvTranspose1d(
                in_channels=d_model,
                out_channels=input_dim,
                kernel_size=temporal_kernel_size,
                stride=temporal_stride,
                padding=0
            )
        elif task == 'forecasting':
            self.forecast_decoder = nn.Linear(d_model, input_dim)
        elif task == 'contrastive':
            self.projection_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model // 2)
            )

    def forward(self, x, mask=None, classification=False):
        # x shape: [batch_size, seq_len, feature_dim]
        batch_size, orig_seq_len, feature_dim = x.shape

        # Apply temporal convolution
        x_orig = x  # Store original input for reconstruction
        x = x.transpose(1, 2)  # [batch_size, feature_dim, seq_len]
        x = self.temporal_tokenizer(x)  # [batch_size, d_model, new_seq_len]
        conv_seq_len = x.size(2)  # Store the sequence length after convolution
        x = x.transpose(1, 2)  # [batch_size, new_seq_len, d_model]

        x = self.positional_encoding(x)

        if mask is not None:
            new_mask = torch.zeros(batch_size, conv_seq_len, dtype=torch.bool, device=x.device)

            for i in range(conv_seq_len):
                start_idx = i * self.temporal_stride
                end_idx = min(start_idx + self.temporal_kernel_size, orig_seq_len)

                for b in range(batch_size):
                    if mask[b, start_idx:end_idx].any():
                        new_mask[b, i] = True

            mask = new_mask

        if mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            x = self.transformer_encoder(x)

        if classification or self.task == 'classification':
            x = torch.mean(x, dim=1)
            return self.classifier(x)

        elif self.task == 'contrastive':
            # Global pooling and projection
            x = torch.mean(x, dim=1)
            return self.projection_head(x)

        elif self.task == 'forecasting':
            # Return sequence of predictions
            return self.forecast_decoder(x)

        elif self.task == 'pretraining':
            x = x.transpose(1, 2)  # [batch_size, d_model, new_seq_len]

            x = self.temporal_decoder(x)  # [batch_size, input_dim, ~orig_seq_len]

            output_seq_len = x.size(2)

            if output_seq_len > orig_seq_len:
                # Trim excess length
                x = x[:, :, :orig_seq_len]
            elif output_seq_len < orig_seq_len:
                # Pad to match original length
                padding = torch.zeros(batch_size, self.input_dim, orig_seq_len - output_seq_len, device=x.device)
                x = torch.cat([x, padding], dim=2)

            x = x.transpose(1, 2)  # [batch_size, orig_seq_len, input_dim]

            return x


###########################################
# 3. DATA AUGMENTATION
###########################################

class IMUDataAugmenter:
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.jitter_scale = config.get('jitter_scale', 0.1)
        self.time_warp_scale = config.get('time_warp_scale', 0.2)
        self.rotation_angle = config.get('rotation_angle', 10)
        self.permutation_segments = config.get('permutation_segments', 3)
        self.magnitude_scale = config.get('magnitude_scale', 0.1)

    def jitter(self, x):
        """Add random noise"""
        return x + torch.randn_like(x) * self.jitter_scale * x.std(dim=1, keepdim=True)

    def scale_magnitude(self, x):
        """Apply random scaling to signal magnitude"""
        factor = torch.randn(x.shape[0], 1, x.shape[2]) * self.magnitude_scale + 1
        factor = factor.to(x.device)
        return x * factor

    def time_warp(self, x):
        """Apply random time warping"""
        batch_size, seq_len, features = x.shape

        warp = torch.zeros(batch_size, seq_len).to(x.device)
        for i in range(batch_size):
            num_pts = 5
            pts_x = torch.linspace(0, seq_len - 1, num_pts)
            pts_y = torch.randn(num_pts) * self.time_warp_scale * seq_len

            for j in range(seq_len):
                right_idx = torch.sum(pts_x <= j).item()
                if right_idx == 0:
                    warp[i, j] = pts_y[0]
                elif right_idx == num_pts:
                    warp[i, j] = pts_y[-1]
                else:
                    left_idx = right_idx - 1
                    alpha = (j - pts_x[left_idx]) / (pts_x[right_idx] - pts_x[left_idx])
                    warp[i, j] = pts_y[left_idx] * (1 - alpha) + pts_y[right_idx] * alpha

        warped_x = torch.zeros_like(x)
        for i in range(batch_size):
            for j in range(seq_len):
                src_idx = min(max(int(j + warp[i, j]), 0), seq_len - 1)
                warped_x[i, j] = x[i, src_idx]

        return warped_x

    def rotate(self, x):
        """Apply rotations to accelerometer and gyroscope data"""
        batch_size = x.shape[0]
        device = x.device

        has_acc = x.shape[2] >= 3
        has_gyro = x.shape[2] >= 6

        if not has_acc:
            return x

        rotation_matrices = []
        for _ in range(batch_size):
            angle_x = np.radians(np.random.uniform(-self.rotation_angle, self.rotation_angle))
            angle_y = np.radians(np.random.uniform(-self.rotation_angle, self.rotation_angle))
            angle_z = np.radians(np.random.uniform(-self.rotation_angle, self.rotation_angle))

            R_x = torch.tensor([
                [1, 0, 0],
                [0, np.cos(angle_x), -np.sin(angle_x)],
                [0, np.sin(angle_x), np.cos(angle_x)]
            ], dtype=torch.float32, device=device)

            R_y = torch.tensor([
                [np.cos(angle_y), 0, np.sin(angle_y)],
                [0, 1, 0],
                [-np.sin(angle_y), 0, np.cos(angle_y)]
            ], dtype=torch.float32, device=device)

            R_z = torch.tensor([
                [np.cos(angle_z), -np.sin(angle_z), 0],
                [np.sin(angle_z), np.cos(angle_z), 0],
                [0, 0, 1]
            ], dtype=torch.float32, device=device)

            R = torch.mm(torch.mm(R_z, R_y), R_x)
            rotation_matrices.append(R)

        rotated_x = x.clone()
        for i in range(batch_size):
            acc_data = x[i, :, 0:3]
            rotated_acc = torch.matmul(acc_data, rotation_matrices[i].T)
            rotated_x[i, :, 0:3] = rotated_acc

            if has_gyro:
                gyro_data = x[i, :, 3:6]
                rotated_gyro = torch.matmul(gyro_data, rotation_matrices[i].T)
                rotated_x[i, :, 3:6] = rotated_gyro

        return rotated_x

    def __call__(self, x):
        """Apply random augmentations"""
        augmentations = [
            self.jitter,
            self.scale_magnitude,
            self.time_warp,
            self.rotate
        ]

        num_augmentations = torch.randint(1, len(augmentations) + 1, (1,)).item()
        chosen_augmentations = torch.randperm(len(augmentations))[:num_augmentations]

        augmented_x = x.clone()
        for aug_idx in chosen_augmentations:
            augmented_x = augmentations[aug_idx](augmented_x)

        return augmented_x


###########################################
# 4. SUPERVISED TRAINING UTILITIES
###########################################

def load_pretrained_model(checkpoint_path, config):
    """
    Load a pretrained model from checkpoint

    Parameters:
    - checkpoint_path: Path to the pretrained model checkpoint
    - config: Configuration dictionary

    Returns:
    - Initialized model with pretrained weights
    """
    pretraining_task = config.get('pretraining_task', 'pretraining')

    model = IMUTransformer(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        max_seq_length=config['window_size'],
        task='classification',
        num_classes=config['num_classes'],
        temporal_kernel_size=config.get('temporal_kernel_size', 8),
        temporal_stride=config.get('temporal_stride', 1)
    )

    if os.path.exists(checkpoint_path):
        print(f"Loading pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config['device'])

        if 'config' in checkpoint:
            pretrain_config = checkpoint['config']
            pretraining_task = pretrain_config.get('task', pretraining_task)
            print(f"Loaded pretrained model was trained on task: {pretraining_task}")

            for key in ['d_model', 'nhead', 'num_layers', 'temporal_kernel_size', 'temporal_stride']:
                if key in pretrain_config:
                    print(f"Using {key}={pretrain_config[key]} from pretrained model")
                    config[key] = pretrain_config[key]

        state_dict = checkpoint['model_state_dict']

        try:
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            print(f"Warning: Could not load some parameters: {e}")

            encoder_dict = {k: v for k, v in state_dict.items()
                            if 'temporal_tokenizer' in k or 'transformer_encoder' in k}
            if encoder_dict:
                print("Loading only encoder parameters")
                model.load_state_dict(encoder_dict, strict=False)
            else:
                print("ERROR: Could not load pretrained weights")

    else:
        print(f"WARNING: Pretrained weights file not found at {checkpoint_path}")
        print("Training will start from random initialization")
        assert os.path.exists(checkpoint_path), f'ckpt does not exists: {checkpoint_path}'

    freeze_strategy = config.get('freeze_strategy', 'encoder')

    if freeze_strategy == 'all':
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    elif freeze_strategy == 'encoder':
        for name, param in model.named_parameters():
            if 'transformer_encoder' in name:
                param.requires_grad = False

    elif freeze_strategy == 'none':
        pass

    else:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Model has {total_params} total parameters")
    print(f"Trainable parameters: {trainable_params} ({trainable_params / total_params:.2%})")

    print("\nTrainable layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel()} parameters")

    return model



def prepare_supervised_data(labeled_files_dir, config):
    """
    Prepare data for supervised training from files containing labeled gestures
    with proper train/test split within each file

    Parameters:
    - labeled_files_dir: Directory containing labeled CSV files
    - config: Configuration dictionary

    Returns:
    - X_train, y_train: Training data and labels
    - X_val, y_val: Validation data and labels
    """
    class_mapping = {}

    train_dir = os.path.join(labeled_files_dir, 'train')
    val_dir = os.path.join(labeled_files_dir, 'test')

    class_names = set()
    for filename in os.listdir(train_dir):
        if filename.startswith('.DS'):
            continue
        class_names.add(filename)
        # if filename.endswith('.csv'):
        #     class_name = os.path.splitext(filename)[0]
        #     class_names.add(class_name)

    for idx, name in enumerate(sorted(class_names)):
        class_mapping[name] = idx

    print(f"Found {len(class_mapping)} classes: {class_mapping}")

    with open(os.path.join(config["output_dir"], "class_mapping.json"), 'w') as f:
        json.dump(class_mapping, f, indent=2)

    train_windows = []
    train_labels = []
    val_windows = []
    val_labels = []

    for file_path in natsorted(glob.glob(os.path.join(train_dir, '*/*.csv'))):
        filename = os.path.basename(file_path)
        class_name = file_path.split('/')[-2]  # os.path.splitext(filename)[0]

        if class_name not in class_mapping:
            print(f"Skipping file {filename} - class not recognized")
            continue

        class_label = class_mapping[class_name]

        data = load_imu_data_from_csv(file_path)
        if data is None or len(data) == 0:
            continue

        print(f"Processing file {filename} with {len(data)} rows for class {class_name} (label {class_label})")

        train_processed, train_indices, _ = preprocess_imu_data(data, config)
        if train_processed is not None:
            train_labels_array = np.full(train_processed.shape[0], class_label)
            train_windows.append(train_processed)
            train_labels.append(train_labels_array)

            train_windows, train_labels = augment_windows(
                train_windows, train_labels, data, class_label, config
            )

    for file_path in natsorted(glob.glob(os.path.join(val_dir, '*/*.csv'))):
        filename = os.path.basename(file_path)
        class_name = file_path.split('/')[-2]  # os.path.splitext(filename)[0]

        if class_name not in class_mapping:
            print(f"Skipping file {filename} - class not recognized")
            continue

        class_label = class_mapping[class_name]

        data = load_imu_data_from_csv(file_path)
        if data is None or len(data) == 0:
            continue

        print(f"Processing file {filename} with {len(data)} rows for class {class_name} (label {class_label})")

        val_processed, val_indices, _ = preprocess_imu_data(data, config)
        if val_processed is not None:
            val_labels_array = np.full(val_processed.shape[0], class_label)
            val_windows.append(val_processed)
            val_labels.append(val_labels_array)

            # val_windows, val_labels = augment_windows(
            #     val_windows, val_labels, data, class_label, config
            # )


    X_train = np.concatenate(train_windows, axis=0) if train_windows else np.array([])
    y_train = np.concatenate(train_labels, axis=0) if train_labels else np.array([])
    X_val = np.concatenate(val_windows, axis=0) if val_windows else np.array([])
    y_val = np.concatenate(val_labels, axis=0) if val_labels else np.array([])

    if config.get('balance_classes', True):
        X_train, y_train = balance_classes(X_train, y_train, config)

    print(f"Final dataset sizes:")
    print(f"  Training set: {len(X_train)} samples")
    for cls, count in sorted({cls: (y_train == cls).sum() for cls in np.unique(y_train)}.items()):
        print(f"    Class {cls}: {count} samples ({count / len(y_train) * 100:.1f}%)")

    print(f"  Testing set: {len(X_val)} samples")
    for cls, count in sorted({cls: (y_val == cls).sum() for cls in np.unique(y_val)}.items()):
        print(f"    Class {cls}: {count} samples ({count / len(y_val) * 100:.1f}%)")

    return X_train, y_train, X_val, y_val


def augment_windows(windows_list, labels_list, raw_data, class_label, config):
    base_window_size = config.get('window_size', 128)

    min_window = int(base_window_size * 0.6)
    max_window = int(base_window_size * 1.4)
    step = (max_window - min_window) // 4

    window_sizes = [min_window + i * step for i in range(5)]
    if base_window_size not in window_sizes:
        window_sizes.append(base_window_size)
    window_sizes = sorted(list(set(window_sizes)))

    # window_sizes = [
    #     # int(base_window_size * 0.8),
    #     base_window_size,
    #     # int(base_window_size * 1.2)
    # ]

    print(f"  Using window sizes for augmentation: {window_sizes}")

    for window_size in window_sizes:
        if window_size == base_window_size:
            continue

        aug_config = config.copy()
        aug_config['window_size'] = window_size
        aug_config['overlap'] = config.get('overlap', 0.75)

        aug_data, _, _ = preprocess_imu_data(raw_data, aug_config)
        # print(aug_data.shape)
        if aug_data is None or aug_data.shape[0] == 0:
            continue

        resized_data = resize_windows(aug_data, window_size, base_window_size)

        aug_labels = np.full(resized_data.shape[0], class_label)
        windows_list.append(resized_data)
        labels_list.append(aug_labels)

    return windows_list, labels_list


def resize_windows(windows, current_size, target_size):
    if current_size == target_size:
        return windows

    resized_data = []

    for window in windows:
        if current_size < target_size:
            x = np.linspace(0, 1, current_size)
            x_new = np.linspace(0, 1, target_size)
            resized_window = np.zeros((target_size, window.shape[1]))

            for i in range(window.shape[1]):
                resized_window[:, i] = np.interp(x_new, x, window[:, i])

        else:  # current_size > target_size
            if np.random.random() < 0.5:
                indices = np.linspace(0, current_size - 1, target_size, dtype=int)
                resized_window = window[indices, :]
            else:
                start_idx = np.random.randint(0, current_size - target_size + 1)
                resized_window = window[start_idx:start_idx + target_size, :]

        resized_data.append(resized_window)

    return np.array(resized_data)


def balance_classes(X, y, config):
    """
    Balance classes in training data through augmentation

    Parameters:
    - X: Feature windows
    - y: Class labels
    - config: Configuration dictionary

    Returns:
    - X_balanced, y_balanced: Balanced data and labels
    """
    class_counts = {}
    for cls in np.unique(y):
        class_counts[cls] = (y == cls).sum()

    max_count = max(class_counts.values())
    min_count = min(class_counts.values())

    if max_count / min_count < config.get('balance_threshold', 1.5):
        print("Class distribution is already balanced. Skipping augmentation.")
        return X, y

    augmenter = IMUDataAugmenter(config)
    balanced_windows = []
    balanced_labels = []
    balanced_windows.append(X)
    balanced_labels.append(y)

    for cls, count in class_counts.items():
        additional_needed = max_count - count

        if additional_needed > 0:
            print(f"  Augmenting class {cls} with {additional_needed} additional samples")
            cls_indices = np.where(y == cls)[0]
            cls_windows = X[cls_indices]

            for _ in range(additional_needed):
                idx = np.random.randint(0, len(cls_windows))
                window = cls_windows[idx]
                tensor_window = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                augmented = augmenter(tensor_window)
                augmented = augmented.squeeze(0).numpy()
                balanced_windows.append(augmented.reshape(1, *augmented.shape))
                balanced_labels.append(np.array([cls]))

    X_balanced = np.concatenate(balanced_windows, axis=0)
    y_balanced = np.concatenate(balanced_labels, axis=0)

    return X_balanced, y_balanced


class SupervisedIMUDataset(Dataset):
    def __init__(self, windows, labels, config=None, augment=False):
        self.windows = windows
        self.labels = labels
        self.augment = augment

        if augment:
            self.augmenter = IMUDataAugmenter(config)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        window = self.windows[idx]
        label = self.labels[idx]

        if self.augment:
            tensor_window = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
            if random.random() < 0.5:
                tensor_window = self.augmenter(tensor_window)

            window = tensor_window.squeeze(0).numpy()

        return torch.tensor(window, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def train_supervised(config):
    """
    Train a supervised model using pretrained weights

    Parameters:
    - config: Configuration dictionary including paths and hyperparameters

    Returns:
    - Trained model and training metrics
    """
    set_seed(config.get('random_seed', 42))
    os.makedirs(config['output_dir'], exist_ok=True)
    config_path = os.path.join(config['output_dir'], 'training_config.json')
    import json
    with open(config_path, 'w') as f:
        json.dump({k: str(v) if not isinstance(v, (int, float, bool, str, list, dict)) else v
                   for k, v in config.items() if k != 'normalization_stat'}, f, indent=2)

    print(f"Configuration saved to {config_path}")
    print("\nPreparing data with proper train/test split at the file level...")
    X_train, y_train, X_val, y_val = prepare_supervised_data(config['labeled_data_dir'], config)
    config['num_classes'] = len(np.unique(np.concatenate([y_train, y_val])))

    print("\n" + "=" * 50)
    print(f"Starting supervised training with {len(X_train)} training samples and {len(X_val)} validation samples")
    print(f"Number of classes: {config['num_classes']}")
    print(f"Input shape: {X_train.shape}")
    print("=" * 50 + "\n")

    print("Class distribution:")
    for cls in sorted(np.unique(np.concatenate([y_train, y_val]))):
        train_count = (y_train == cls).sum()
        val_count = (y_val == cls).sum()
        print(
            f"Class {cls}: {train_count} train ({train_count / len(y_train) * 100:.1f}%), "
            f"{val_count} val ({val_count / len(y_val) * 100:.1f}%)")

    train_dataset = SupervisedIMUDataset(X_train, y_train, config=config, augment=True)
    val_dataset = SupervisedIMUDataset(X_val, y_val, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False)
    )
    model = load_pretrained_model(config['pretrained_path'], config)
    param_groups = []

    if config.get('use_layerwise_lr', False):
        classifier_params = [p for n, p in model.named_parameters()
                             if 'classifier' in n and p.requires_grad]
        if classifier_params:
            param_groups.append({
                'params': classifier_params,
                'lr': config['learning_rate']
            })

        encoder_params = [p for n, p in model.named_parameters()
                          if 'transformer_encoder' in n and p.requires_grad]
        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': config['learning_rate'] * 0.1
            })

        other_params = [p for n, p in model.named_parameters()
                        if 'classifier' not in n and 'transformer_encoder' not in n and p.requires_grad]
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': config['learning_rate'] * 0.5
            })
    else:
        param_groups.append({
            'params': [p for p in model.parameters() if p.requires_grad],
            'lr': config['learning_rate']
        })

    optimizer_type = config.get('optimizer', 'adam').lower()

    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 1e-5),
            nesterov=config.get('nesterov', True)
        )
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=config.get('weight_decay', 1e-5)
        )
    else: 
        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=config.get('weight_decay', 1e-5)
        )

    scheduler_type = config.get('scheduler', 'plateau').lower()

    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'],
            eta_min=config.get('min_lr', 1e-6)
        )
    elif scheduler_type == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['learning_rate'],
            steps_per_epoch=len(train_loader),
            epochs=config['num_epochs']
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('lr_factor', 0.5),
            patience=config.get('patience', 5),
            min_lr=config.get('min_lr', 1e-6),
            verbose=True
        )

    if config.get('use_class_weights', False):
        class_counts = np.bincount(y_train.astype(int))
        class_weights = 1.0 / class_counts
        class_weights = class_weights / np.sum(class_weights) * len(class_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=config['device'])
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using weighted loss with class weights: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()

    model.to(config['device'])
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    lr_history = []
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0

    patience = config.get('early_stopping_patience', 15)
    patience_counter = 0

    print("\nStarting training...")
    for epoch in range(config['num_epochs']):
        start_time = time.time()

        if config.get('progressive_unfreezing', False):
            if epoch == config.get('unfreeze_after', 5):
                print("Unfreezing transformer encoder layers...")
                for name, param in model.named_parameters():
                    if 'transformer_encoder' in name:
                        param.requires_grad = True

                param_groups = []
                if config.get('use_layerwise_lr', False):
                    param_groups = [
                        {'params': [p for n, p in model.named_parameters()
                                    if 'classifier' in n and p.requires_grad],
                         'lr': config['learning_rate']},
                        {'params': [p for n, p in model.named_parameters()
                                    if 'transformer_encoder' in n and p.requires_grad],
                         'lr': config['learning_rate'] * 0.1},
                        {'params': [p for n, p in model.named_parameters()
                                    if 'classifier' not in n and 'transformer_encoder' not in n and p.requires_grad],
                         'lr': config['learning_rate'] * 0.5}
                    ]
                    param_groups = [g for g in param_groups if len(g['params']) > 0]
                else:
                    param_groups = [{'params': [p for p in model.parameters() if p.requires_grad]}]

                if optimizer_type == 'sgd':
                    optimizer = torch.optim.SGD(
                        param_groups,
                        lr=config['learning_rate'] * 0.1,
                        momentum=config.get('momentum', 0.9),
                        weight_decay=config.get('weight_decay', 1e-5),
                        nesterov=config.get('nesterov', True)
                    )
                elif optimizer_type == 'adamw':
                    optimizer = torch.optim.AdamW(
                        param_groups,
                        lr=config['learning_rate'] * 0.1,
                        weight_decay=config.get('weight_decay', 1e-5)
                    )
                else: 
                    optimizer = torch.optim.Adam(
                        param_groups,
                        lr=config['learning_rate'] * 0.1,
                        weight_decay=config.get('weight_decay', 1e-5)
                    )

                if scheduler_type == 'cosine':
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=config['num_epochs'] - epoch,
                        eta_min=config.get('min_lr', 1e-6)
                    )
                elif scheduler_type == 'onecycle':
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=config['learning_rate'] * 0.1,
                        steps_per_epoch=len(train_loader),
                        epochs=config['num_epochs'] - epoch
                    )
                else:
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode='min',
                        factor=config.get('lr_factor', 0.5),
                        patience=config.get('patience', 5),
                        min_lr=config.get('min_lr', 1e-6),
                        verbose=True
                    )

        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config['device']), target.to(config['device'])

            optimizer.zero_grad()
            output = model(data, classification=True)
            loss = criterion(output, target)
            loss.backward()

            if config.get('gradient_clipping', True):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('clip_value', 1.0))

            optimizer.step()

            if scheduler_type == 'onecycle':
                scheduler.step()

            total_train_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

            if (batch_idx + 1) % config.get('print_freq', 10) == 0:
                print(f'Epoch {epoch + 1}/{config["num_epochs"]} '
                      f'[{batch_idx + 1}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} '
                      f'Acc: {100 * train_correct / train_total:.2f}%')

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        train_accuracy = train_correct / train_total
        train_accuracies.append(train_accuracy)

        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0

        all_preds = []
        all_targets = []

        class_correct = {}
        class_total = {}

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(config['device']), target.to(config['device'])
                output = model(data, classification=True)
                val_loss = criterion(output, target)
                total_val_loss += val_loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                for c in range(config['num_classes']):
                    mask = (target == c)
                    class_total[c] = class_total.get(c, 0) + mask.sum().item()
                    class_correct[c] = class_correct.get(c, 0) + ((predicted == target) & mask).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        val_accuracy = val_correct / val_total
        val_accuracies.append(val_accuracy)
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        if scheduler_type == 'plateau':
            scheduler.step(avg_val_loss)
        elif scheduler_type == 'cosine':
            scheduler.step()

        epoch_time = time.time() - start_time

        # class_accs = []

        print("\nPer-class validation accuracy:")
        for c in range(config['num_classes']):
            if class_total[c] > 0:
                acc = 100 * class_correct[c] / class_total[c]
                # class_accs.append(acc)
                print(f"  Class {c}: {acc:.2f}% ({class_correct[c]}/{class_total[c]})")

        # val_accuracy = (sum(class_accs) / len(class_accs))/100.0
        # val_accuracies.append(val_accuracy)

        print(f'\nEpoch {epoch + 1} summary:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy * 100:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy * 100:.2f}%')
        print(f'  Learning Rate: {current_lr:.1e}, Time: {epoch_time:.1f}s')

        improved = False

        if val_accuracy > best_val_acc or (val_accuracy == best_val_acc and avg_val_loss < best_val_loss):
            improved = True
            best_val_acc = val_accuracy
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            best_preds = all_preds
            best_targets = all_targets

            print(f"  New best model saved! (Val Acc: {val_accuracy * 100:.2f}%, Val Loss: {avg_val_loss:.4f})")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'train_accuracy': train_accuracy,
                'config': config,
            }, os.path.join(config['output_dir'], 'best_supervised_model.pth'))

        if improved:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break

        print(f"  Patience: {patience_counter}/{patience}")
        print("-" * 40)

    print("\nTraining complete!")
    print(f"Best model at epoch {best_epoch + 1} with validation accuracy: {best_val_acc * 100:.2f}%")

    plot_training_metrics(train_losses, val_losses, train_accuracies,
                          val_accuracies, lr_history, best_epoch, config)

    plot_confusion_matrix(best_targets, best_preds, config)

    torch.save({
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'best_model_state_dict': best_model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'config': config,
    }, os.path.join(config['output_dir'], 'final_supervised_model.pth'))

    model.load_state_dict(best_model_state)

    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'lr_history': lr_history,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss
    }


def plot_training_metrics(train_losses, val_losses, train_accuracies,
                          val_accuracies, lr_history, best_epoch, config):
    """
    Plot and save training metrics

    Parameters:
    - train_losses, val_losses: Lists of training and validation losses
    - train_accuracies, val_accuracies: Lists of training and validation accuracies
    - lr_history: List of learning rates
    - best_epoch: Epoch with best performance
    - config: Configuration dictionary
    """
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Model (Epoch {best_epoch + 1})')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.axvline(x=best_epoch, color='r', linestyle='--')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.semilogy(train_losses, label='Train Loss')
    plt.semilogy(val_losses, label='Validation Loss')
    plt.axvline(x=best_epoch, color='r', linestyle='--')
    plt.title('Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(lr_history)
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    if len(lr_history) > 0:
        plt.yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], 'supervised_training_metrics.png'))
    plt.close()


def plot_confusion_matrix(targets, predictions, config):
    """
    Plot and save confusion matrix

    Parameters:
    - targets: True class labels
    - predictions: Predicted class labels
    - config: Configuration dictionary
    """
    try:
        from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
        import matplotlib.pyplot as plt

        cm = confusion_matrix(targets, predictions)

        report = classification_report(targets, predictions, digits=3)

        with open(os.path.join(config['output_dir'], 'classification_report.txt'), 'w') as f:
            f.write(report)

        print("\nClassification Report:")
        print(report)

        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(config['output_dir'], 'confusion_matrix.png'))
        plt.close()

    except ImportError:
        print("Couldn't create confusion matrix plot. sklearn might be missing.")


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compare_pretraining_methods(config, pretrained_paths):
    """
    Compare different pretraining methods for the same supervised task

    Parameters:
    - config: Base configuration dictionary
    - pretrained_paths: Dictionary mapping method names to pretrained model paths

    Example:
    pretrained_paths = {
        'masked': 'output_masked/best_pretrained_model.pth',
        'transition': 'output_transition/best_transition_detection_model.pth',
        'primitive': 'output_primitive/best_primitive_identification_model.pth'
    }
    """
    results = {}

    comparison_dir = os.path.join(config['output_dir'], 'pretraining_comparison')
    os.makedirs(comparison_dir, exist_ok=True)

    for method_name, model_path in pretrained_paths.items():
        print(f"\n{'=' * 50}")
        print(f"Evaluating pretraining method: {method_name}")
        print(f"{'=' * 50}")

        method_config = config.copy()
        method_config['pretrained_path'] = model_path
        method_config['pretraining_task'] = method_name
        method_config['output_dir'] = os.path.join(comparison_dir, method_name)

        model, metrics = train_supervised(method_config)

        results[method_name] = metrics

    compare_training_curves(results, comparison_dir)

    return results


def compare_training_curves(results, output_dir):
    """
    Create comparison plots for different pretraining methods

    Parameters:
    - results: Dictionary mapping method names to training metrics
    - output_dir: Directory to save comparison plots
    """
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    for method, metrics in results.items():
        plt.plot(metrics['val_accuracies'], label=f"{method}")
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    for method, metrics in results.items():
        plt.plot(metrics['val_losses'], label=f"{method}")
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    for method, metrics in results.items():
        plt.plot(metrics['train_accuracies'], label=f"{method}")
    plt.title('Training Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    methods = list(results.keys())
    best_accs = [metrics['best_val_acc'] * 100 for metrics in results.values()]

    sorted_indices = np.argsort(best_accs)[::-1]  # Descending order
    sorted_methods = [methods[i] for i in sorted_indices]
    sorted_accs = [best_accs[i] for i in sorted_indices]

    bars = plt.bar(sorted_methods, sorted_accs)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.2f}%', ha='center', va='bottom')

    plt.title('Best Validation Accuracy Comparison')
    plt.xlabel('Pretraining Method')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pretraining_comparison.png'))
    plt.close()

    summary = ["Pretraining Method Comparison Summary"]
    summary.append("=" * 80)
    summary.append(f"{'Method':<20} {'Best Val Acc':<15} {'Best Val Loss':<15} {'Best Epoch':<10}")
    summary.append("-" * 80)

    for method in sorted_methods:
        metrics = results[method]
        summary.append(
            f"{method:<20} {metrics['best_val_acc'] * 100:>13.2f}% {metrics['best_val_loss']:>14.4f} {metrics['best_epoch'] + 1:>10}")

    with open(os.path.join(output_dir, 'comparison_summary.txt'), 'w') as f:
        f.write('\n'.join(summary))

    print("\nPretraining Method Comparison:")
    for line in summary:
        print(line)


def compute_and_save_normalization_stats(data_dir, output_dir, features=None):
    """
    Load all CSV files from the given directory, compute normalization statistics,
    and save them for later use.

    Parameters:
    - data_dir: Directory containing CSV files or subdirectories with CSV files
    - output_dir: Directory to save the normalization statistics
    - features: List of feature column names to normalize (default: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)

    Returns:
    - Dictionary containing the normalization statistics (mean and std)
    """

    if features is None:
        features = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

    csv_files = glob.glob(os.path.join(data_dir, '*/*/*.csv'))
    # csv_files.extend(glob.glob(os.path.join(data_dir, '*/*.csv')))
    # csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    # csv_files.extend(glob.glob(os.path.join(data_dir, '*/*.csv')))

    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    print(f"Found {len(csv_files)} CSV files")

    all_feature_data = []

    for csv_file in csv_files:
        try:
            data = pd.read_csv(csv_file)
            if not all(feature in data.columns for feature in features):
                missing = [f for f in features if f not in data.columns]
                print(f"Warning: File {csv_file} missing features: {missing}. Skipping.")
                continue

            feature_data = data[features].values
            all_feature_data.append(feature_data)

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    if not all_feature_data:
        raise ValueError("No valid data found in CSV files")

    combined_data = np.vstack(all_feature_data)
    print(f"Combined data shape: {combined_data.shape}")

    scaler = StandardScaler()
    scaler.fit(combined_data)
    mean_values = scaler.mean_
    std_values = scaler.scale_
    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, 'normalization_stats.npz')
    np.savez(stats_path, mean=mean_values, std=std_values, feature_names=features)

    print(f"Normalization statistics saved to {stats_path}")
    stats = {
        'mean': mean_values,
        'std': std_values,
        'feature_names': features
    }

    return stats


if __name__ == "__main__":
    import time
    users = ['P03', 'P04', 'P05', 'P06']
    for user in users:
        BASE_CONFIG = {
            'data_dir': f'./dataset/{user}',  # Directory containing CSV files
            'output_dir': f'./output/{user}',  # Base directory for outputs
            'input_dim': 6,  # Number of input channels (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
            'window_size': 120,  # Number of samples in each window
            'overlap': 0.9,  # Fraction of overlap between consecutive windows
            'batch_size': 32,  # Batch size for training
            'num_epochs': 50,  # Maximum number of training epochs
            'd_model': 128,  # Model dimension
            'nhead': 4,  # Number of attention heads
            'num_layers': 3,  # Number of transformer layers
            'dropout': 0.2,  # Dropout rate
            'device': 'cuda:3' if torch.cuda.is_available() else 'cpu',  # Device for training
            'learning_rate': 5e-4,  # Base learning rate for supervised training
            'weight_decay': 1e-5,  # Weight decay for regularization
            'scheduler': 'plateau',  # Learning rate scheduler ('plateau', 'cosine', or 'onecycle')
            'lr_factor': 0.5,  # Factor by which to reduce learning rate on plateau
            'patience': 5,  # Number of epochs to wait before reducing learning rate
            'early_stopping_patience': 8,  # Patience for early stopping
            'sampling_rate': 50,  # IMU data sampling rate in Hz
            'labeled_data_dir': f'./dataset/{user}',  # Directory with class-labeled files
            'gradient_clipping': True,  # Whether to use gradient clipping
            'clip_value': 1.0,  # Maximum gradient norm
            'use_class_weights': True,  # Whether to use class weights in loss function
            'balance_classes': True,  # Whether to balance classes through augmentation
            'balance_threshold': 1.5,  # Minimum imbalance ratio to trigger class balancing
            'jitter_scale': 0.1,  # Scale of jitter noise
            'time_warp_scale': 0.2,  # Scale of time warping
            'rotation_angle': 10,  # Maximum rotation angle in degrees
            'magnitude_scale': 0.1,  # Scale for magnitude scaling
            'random_seed': 42,  # Random seed for reproducibility
            'num_workers': 4,  # Number of dataloader workers
            'pin_memory': True,  # Whether to pin memory in dataloader
            'progressive_unfreezing': True,  # Whether to progressively unfreeze layers
            'unfreeze_after': 5,  # Epoch to start unfreezing encoder
            'use_layerwise_lr': True,  # Whether to use different learning rates for different layers
        }

        norm_stat = compute_and_save_normalization_stats(
            BASE_CONFIG['data_dir'],
            BASE_CONFIG['output_dir']
        )

        BASE_CONFIG['normalization_stat'] = norm_stat
        def compare_pretraining_methods_():
            pretrained_paths = {
                'cpc': f"./pretrained_models/{user}/best_cpc_model.pth",
                'contrastive': f"./pretrained_models/{user}/best_contrastive_model.pth",
                'masked_modeling': f'./pretrained_models/{user}__/best_pretrained_imu_transformer.pth',
            }

            config = BASE_CONFIG.copy()
            config['num_epochs'] = 25
            config['early_stopping_patience'] = 8
            config['output_dir'] = f'./output/{user}'
            config['freeze_strategy'] = 'none'
            start_time = time.time()
            results = compare_pretraining_methods(config, pretrained_paths)
            comparison_time = time.time() - start_time
            best_method = max(results.items(), key=lambda x: x[1]['best_val_acc'])[0]
            best_acc = results[best_method]['best_val_acc'] * 100
            print(f"Best pretraining method: {best_method} with validation accuracy: {best_acc:.2f}%")
            return results

        results = compare_pretraining_methods_()
        