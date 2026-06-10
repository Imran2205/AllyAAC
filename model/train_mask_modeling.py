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
    overlap = config.get('overlap', 0.5)
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

        if end_idx - start_idx < window_size * 0.5:
            continue

        window = data.iloc[start_idx:end_idx].copy()

        if end_idx - start_idx < window_size:
            pad_size = window_size - (end_idx - start_idx)
            padding = pd.DataFrame(0, index=range(pad_size), columns=data.columns)
            window = pd.concat([window, padding], ignore_index=True)

        windows.append(window)
        window_indices.append((start_idx, end_idx))

    if len(windows) == 0:
        return None, None, None

    windowed_data = []
    imu_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

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


def load_all_sessions(config, split=None):
    """
    Load all CSV files from the data directory

    Parameters:
    - config: Dictionary containing configuration parameters

    Returns:
    - List of preprocessed data from all sessions
    """
    if split is not None:
        data_dir = os.path.join(config.get('data_dir', 'imu_data'), split)
        csv_files = natsorted(glob.glob(os.path.join(data_dir, '*/*.csv')))
    else:
        data_dir = config.get('data_dir', 'imu_data')
        csv_files = natsorted(glob.glob(os.path.join(data_dir, '*.csv')))
    all_data = []

    for csv_file in csv_files:
        print(f"Processing: {csv_file}")
        data = load_imu_data_from_csv(csv_file)
        if data is not None and len(data) > 0:
            preprocessed_data, _, _ = preprocess_imu_data(data, config)
            if preprocessed_data is not None:
                all_data.append(preprocessed_data)

    return all_data


class IMUDataset(Dataset):

    def __init__(self, data_list):
        """
        Initialize the dataset

        Parameters:
        - data_list: List of preprocessed IMU data arrays from different sessions
        """
        self.data = np.concatenate(data_list, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)



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

    def forward(self, x, mask=None, classification=False):
        batch_size, orig_seq_len, feature_dim = x.shape

        x_orig = x
        x = x.transpose(1, 2)
        x = self.temporal_tokenizer(x)
        conv_seq_len = x.size(2)
        x = x.transpose(1, 2)

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

        if self.task == 'classification' or classification:
            x = torch.mean(x, dim=1)
            return self.classifier(x)

        elif self.task == 'pretraining':
            x = x.transpose(1, 2)

            x = self.temporal_decoder(x)

            output_seq_len = x.size(2)

            if output_seq_len > orig_seq_len:
                x = x[:, :, :orig_seq_len]
            elif output_seq_len < orig_seq_len:
                padding = torch.zeros(batch_size, self.input_dim, orig_seq_len - output_seq_len, device=x.device)
                x = torch.cat([x, padding], dim=2)

            x = x.transpose(1, 2)
            assert x.shape[0] == batch_size

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
        return x + torch.randn_like(x) * self.jitter_scale * x.std(dim=1, keepdim=True)

    def scale_magnitude(self, x):
        factor = torch.randn(x.shape[0], 1, x.shape[2]) * self.magnitude_scale + 1
        factor = factor.to(x.device)
        return x * factor

    def time_warp(self, x):
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
# 4. SELF-SUPERVISED TRAINING
###########################################

class MaskedIMUModelPretraining:
    def __init__(self, model, config=None, device=None):
        if config is None:
            config = {}

        if device is None:
            device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model
        self.device = device
        self.model.to(device)

        lr = config.get('learning_rate', 1e-4)
        factor = config.get('lr_factor', 0.5)
        patience = config.get('patience', 5)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=factor, patience=patience
        )
        self.augmenter = IMUDataAugmenter(config)

    def create_masked_samples(self, x, mask_ratio=0.15):
        batch_size, seq_len, feature_dim = x.shape

        mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        for i in range(batch_size):
            num_masks = int(seq_len * mask_ratio)
            mask_indices = torch.randperm(seq_len)[:num_masks]
            mask[i, mask_indices] = True

        masked_x = x.clone()
        masked_x[mask.unsqueeze(-1).expand_as(x)] = 0

        return masked_x, mask

    def train_epoch(self, dataloader, config=None):
        if config is None:
            config = {}

        mask_ratio = config.get('mask_ratio', 0.15)

        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            if isinstance(batch, list) and len(batch) == 1:
                x = batch[0].to(self.device)
            else:
                x = batch.to(self.device)

            x_augmented = self.augmenter(x)

            masked_x, mask = self.create_masked_samples(x_augmented, mask_ratio)
            masked_x = masked_x.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(masked_x)

            loss = F.mse_loss(
                outputs[mask.unsqueeze(-1).expand_as(outputs)],
                x_augmented[mask.unsqueeze(-1).expand_as(x_augmented)]
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        self.scheduler.step(avg_loss)

        return avg_loss

    def evaluate(self, dataloader, config=None):
        if config is None:
            config = {}

        mask_ratio = config.get('mask_ratio', 0.15)

        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, list) and len(batch) == 1:
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)

                masked_x, mask = self.create_masked_samples(x, mask_ratio)
                masked_x = masked_x.to(self.device)

                outputs = self.model(masked_x)

                loss = F.mse_loss(
                    outputs[mask.unsqueeze(-1).expand_as(outputs)],
                    x[mask.unsqueeze(-1).expand_as(x)]
                )

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def contrastive_loss(self, z1, z2, temperature=0.5):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        batch_size = z1.size(0)
        features = torch.cat([z1, z2], dim=0)
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / temperature
        return F.cross_entropy(logits, labels)


def train_self_supervised(config=None):
    """
    Run the complete self-supervised training pipeline

    Parameters:
    - config: Dictionary containing configuration parameters including:
        - data_dir: Directory containing CSV files with IMU data
        - output_dir: Directory to save trained models
        - window_size: Number of samples in each window
        - overlap: Fraction of overlap between consecutive windows
        - batch_size: Batch size for training
        - num_epochs: Number of training epochs
        - d_model: Model dimension
        - nhead: Number of attention heads
        - num_layers: Number of transformer layers
        - dropout: Dropout rate
        - device: Device to use for training ('cuda' or 'cpu')
    """
    if config is None:
        config = {}

    config['data_dir'] = config.get('data_dir', 'imu_data')
    config['output_dir'] = config.get('output_dir', 'models')
    config['window_size'] = config.get('window_size', 128)
    config['overlap'] = config.get('overlap', 0.5)
    config['batch_size'] = config.get('batch_size', 32)
    config['num_epochs'] = config.get('num_epochs', 50)
    config['d_model'] = config.get('d_model', 128)
    config['nhead'] = config.get('nhead', 4)
    config['num_layers'] = config.get('num_layers', 3)
    config['dropout'] = config.get('dropout', 0.1)
    config['device'] = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    config['mask_ratio'] = config.get('mask_ratio', 0.15)
    config['learning_rate'] = config.get('learning_rate', 1e-4)
    config['lr_factor'] = config.get('lr_factor', 0.5)
    config['patience'] = config.get('patience', 5)
    config['sampling_rate'] = config.get('sampling_rate', 50)
    config['val_split'] = config.get('val_split', 0.1)
    config['jitter_scale'] = config.get('jitter_scale', 0.1)
    config['time_warp_scale'] = config.get('time_warp_scale', 0.2)
    config['rotation_angle'] = config.get('rotation_angle', 10)
    config['permutation_segments'] = config.get('permutation_segments', 3)
    config['magnitude_scale'] = config.get('magnitude_scale', 0.1)
    config['random_seed'] = config.get('random_seed', 42)

    set_seed(config.get('random_seed', 42))

    device = config['device']
    print(f"Using device: {device}")

    os.makedirs(config['output_dir'], exist_ok=True)


    train_data = load_all_sessions(config, split='train')
    val_data = load_all_sessions(config, split='test')

    if not train_data:
        print("No data found. Exiting.")
        return

    input_dim = train_data[0].shape[-1]
    print(f"Input dimension: {input_dim}")

    config['input_dim'] = input_dim

    train_dataset = IMUDataset(train_data)
    train_dataset_size = len(train_dataset)
    val_dataset = IMUDataset(val_data)
    val_dataset_size = len(val_dataset)
    dataset_size = train_dataset_size + val_dataset_size
    print(f"Dataset size: {dataset_size}")

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"Training dataset size: {len(train_dataloader)}, Val dataset size: {len(val_dataloader)}")

    print("Creating model...")
    model = IMUTransformer(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        max_seq_length=config['window_size'],
        task='pretraining'
    )

    trainer = MaskedIMUModelPretraining(model, config=config)

    print("Starting training...")
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []

    for epoch in range(config['num_epochs']):
        train_loss = trainer.train_epoch(train_dataloader, config=config)

        val_loss = trainer.evaluate(val_dataloader, config=config)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{config['num_epochs']}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(config['output_dir'], 'best_pretrained_masked.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, model_path)
            print(f"Saved best model to {model_path}")

    final_model_path = os.path.join(config['output_dir'], 'final_pretrained_masked.pth')
    torch.save({
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'val_loss': val_loss,
        'config': config,
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, config['num_epochs'] + 1), train_losses, label='Training Loss', marker='o', linestyle='-',
             linewidth=2)
    plt.plot(range(1, config['num_epochs'] + 1), val_losses, label='Validation Loss', marker='x', linestyle='-',
             linewidth=2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Over Time', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()

    loss_plot_path = os.path.join(config['output_dir'], 'loss_curves.png')
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()

    print(f"Loss curves saved to {loss_plot_path}")

    return model

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    release_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(release_root)
    config_root = os.path.join(release_root, 'config', 'pretraining')
    dataset_root = os.path.join(release_root, 'dataset')
    users = sorted(
        user for user in os.listdir(dataset_root)
        if os.path.isfile(os.path.join(config_root, user, 'masked_modeling_config.json'))
    )

    for user in users:
        config_path = os.path.join(config_root, user, 'masked_modeling_config.json')
        with open(config_path, 'r') as f:
            CONFIG = json.load(f)

        CONFIG['data_dir'] = f'./dataset/{user}'
        CONFIG['output_dir'] = f'./pretrained_masked_models/{user}'
        if not torch.cuda.is_available():
            CONFIG['device'] = 'cpu'

        norm_stat = compute_and_save_normalization_stats(
            CONFIG['data_dir'],
            CONFIG['output_dir']
        )

        CONFIG['normalization_stat'] = norm_stat

        model = train_self_supervised(config=CONFIG)
