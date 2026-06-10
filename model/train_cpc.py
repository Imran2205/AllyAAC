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


def preprocess_imu_data_for_forecasting(data, config):
    """
    Preprocess raw IMU data for time series forecasting
    Creates input-target pairs where target is future timestamps

    Parameters:
    - data: DataFrame with IMU data
    - config: Configuration dictionary

    Returns:
    - input_windows: Input sequences for forecasting
    - target_windows: Target future sequences
    - window_indices: Start and end indices for each window
    - scaler: The scaler used for normalization
    """
    window_size = config.get('context_window_size', 80)
    forecast_horizon = config.get('forecast_horizon', 40)
    stride = config.get('stride', 10)
    sampling_rate = config.get('sampling_rate', 50)

    b, a = signal.butter(3, 20 / (sampling_rate / 2), 'low')
    for col in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
        if col in data.columns:
            data[col] = signal.filtfilt(b, a, data[col])

    input_windows = []
    target_windows = []
    window_indices = []

    for start_idx in range(0, len(data) - window_size - forecast_horizon + 1, stride):
        end_idx = start_idx + window_size
        target_end_idx = end_idx + forecast_horizon

        if target_end_idx > len(data):
            break

        input_window = data.iloc[start_idx:end_idx].copy()
        target_window = data.iloc[end_idx:target_end_idx].copy()

        window_indices.append((start_idx, end_idx, target_end_idx))
        input_windows.append(input_window)
        target_windows.append(target_window)

    if len(input_windows) == 0:
        return None, None, None

    imu_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

    input_data = []
    target_data = []

    for i_window, t_window in zip(input_windows, target_windows):
        input_data.append(i_window[imu_columns].values)
        target_data.append(t_window[imu_columns].values)

    input_data = np.array(input_data)
    target_data = np.array(target_data)

    original_input_shape = input_data.shape
    input_data_2d = input_data.reshape(-1, input_data.shape[-1])

    if 'normalization_stat' in config and config['normalization_stat'] is not None:
        mean = config['normalization_stat']['mean']
        std = config['normalization_stat']['std']

        normalized_input_2d = (input_data_2d - mean) / std

        target_data_2d = target_data.reshape(-1, target_data.shape[-1])
        normalized_target_2d = (target_data_2d - mean) / std
    else:
        print("WARNING: Cannot find normalization stat.")
        scaler = StandardScaler()
        normalized_input_2d = scaler.fit_transform(input_data_2d)

        target_data_2d = target_data.reshape(-1, target_data.shape[-1])
        normalized_target_2d = scaler.transform(target_data_2d)

    normalized_input = normalized_input_2d.reshape(original_input_shape)
    normalized_target = normalized_target_2d.reshape(target_data.shape)

    return normalized_input, normalized_target, window_indices


def load_all_sessions_for_forecasting(config, split=None):
    """
    Load all CSV files from the data directory for forecasting task

    Parameters:
    - config: Dictionary containing configuration parameters

    Returns:
    - List of (input_data, target_data) tuples
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
            input_data, target_data, _ = preprocess_imu_data_for_forecasting(data, config)

            if input_data is not None and target_data is not None:
                print(f"  Created {len(input_data)} input-target pairs")
                all_data.append((input_data, target_data))

    return all_data


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


class CPCDataset(Dataset):
    def __init__(self, data_list):
        inputs = [x for (x, _) in data_list]
        self.input_data = np.concatenate(inputs, axis=0)
        print(f"CPC dataset with {len(self.input_data)} samples, input shape: {self.input_data.shape}")

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return torch.tensor(self.input_data[idx], dtype=torch.float32)

class CPC(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3,
                 K=4, max_seq_length=128, temporal_kernel_size=8, temporal_stride=1, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.K = K
        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_stride = temporal_stride

        self.temporal_tokenizer = nn.Conv1d(
            in_channels=input_dim,
            out_channels=d_model,
            kernel_size=temporal_kernel_size,
            stride=temporal_stride,
            padding=0
        )
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers)

        self.g_ar = nn.LSTM(d_model, d_model, num_layers=2, bidirectional=False, batch_first=True)

        self.Wk = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            ) for _ in range(K)
        ])

    def forward(self, x):
        x = x.transpose(1, 2)
        z = self.temporal_tokenizer(x).transpose(1, 2)
        z = self.positional_encoding(z)
        z = self.transformer_encoder(z)
        c, _ = self.g_ar(z)
        return z, c


def cpc_infonce_loss(model, z, c, temperature=0.1, hard_negatives=True):
    B, T, D = z.shape
    K = model.K
    loss = 0.0

    for k in range(1, K + 1):
        valid_t_range = T - K

        t_indices = torch.randint(0, valid_t_range, (B,), device=z.device)

        c_t = c[torch.arange(B), t_indices]
        z_tk = z[torch.arange(B), t_indices + k]

        p_tk = model.Wk[k-1](c_t)
        p_tk = F.normalize(p_tk, dim=-1)
        z_tk = F.normalize(z_tk, dim=-1)

        negatives = []

        negatives.append(z_tk)

        for b in range(B):
            neg_t = torch.randint(0, T, (3,), device=z.device)
            temporal_negs = z[b, neg_t]
            negatives.append(temporal_negs)

        all_targets = torch.cat(negatives, dim=0)
        all_targets = F.normalize(all_targets, dim=-1)

        logits = (p_tk @ all_targets.t()) / temperature
        labels = torch.arange(B, device=z.device)

        loss += F.cross_entropy(logits, labels)

    return loss / K



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

    def __call__(self, x, augment_targets=False):
        """
        Apply random augmentations to input data

        Parameters:
        - x: Input tensor [batch_size, seq_len, feature_dim]
        - augment_targets: Whether to apply same augmentation to target data

        Returns:
        - Augmented tensor
        """
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
# 4. TRAINING
###########################################

class CPCTrainer:
    def __init__(self, model, config=None, device=None):
        if config is None: config = {}
        if device is None:
            device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(device)
        self.device = device

        lr = config.get('learning_rate', 1e-4)
        wd = config.get('weight_decay', 1e-5)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=config.get('lr_factor', 0.5), patience=config.get('patience', 5)
        )

    def train_epoch(self, dataloader):
        self.model.train()
        total = 0.0; n = 0
        for x in dataloader:
            x = x.to(self.device)
            z, c = self.model(x)
            loss = cpc_infonce_loss(self.model, z, c)
            self.optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item(); n += 1
        return total / max(1, n)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total = 0.0; n = 0
        for x in dataloader:
            x = x.to(self.device)
            z, c = self.model(x)
            loss = cpc_infonce_loss(self.model, z, c)
            total += loss.item(); n += 1
        return total / max(1, n)


def train_cpc(config):
    set_seed(config.get('random_seed', 42))
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config['output_dir'], exist_ok=True)

    print("Loading data for CPC...")
    train_data_list = load_all_sessions_for_forecasting(config, split='train')
    val_data_list   = load_all_sessions_for_forecasting(config, split='test')

    if not train_data_list:
        print("No data found."); return

    input_dim = train_data_list[0][0].shape[-1]
    config['input_dim'] = input_dim
    K = config.get('forecast_horizon', 16)

    train_ds = CPCDataset(train_data_list)
    val_ds   = CPCDataset(val_data_list)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config.get('num_workers', 0), pin_memory=config.get('pin_memory', False))
    val_loader   = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False,
                              num_workers=config.get('num_workers', 0), pin_memory=config.get('pin_memory', False))

    model = CPC(
        input_dim=input_dim,
        d_model=config.get('d_model', 128),
        nhead=config.get('nhead', 4),
        num_layers=config.get('num_layers', 3),
        K=K,
        max_seq_length=config.get('context_window_size', 80) + K,
        temporal_kernel_size=config.get('temporal_kernel_size', 8),
        temporal_stride=config.get('temporal_stride', 1),
        dropout=config.get('dropout', 0.1)
    )
    trainer = CPCTrainer(model, config=config, device=device)

    best_val = float('inf'); best_epoch = 0; patience = config.get('early_stopping_patience', 10); counter = 0
    for epoch in range(config.get('num_epochs', 50)):
        tr = trainer.train_epoch(train_loader)
        va = trainer.evaluate(val_loader)
        trainer.scheduler.step(va)
        print(f"Epoch {epoch+1}: train {tr:.4f}  val {va:.4f}")

        if va < best_val:
            best_val = va; best_epoch = epoch; counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config
            }, os.path.join(config['output_dir'], 'best_cpc_model.pth'))
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (best at {best_epoch+1})")
                break

    print(f"Best val loss {best_val:.4f} at epoch {best_epoch+1}")
    return model


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


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    release_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(release_root)
    config_root = os.path.join(release_root, 'config', 'pretraining')
    dataset_root = os.path.join(release_root, 'dataset')
    users = sorted(
        user for user in os.listdir(dataset_root)
        if os.path.isfile(os.path.join(config_root, user, 'cpc_config.json'))
    )

    for user in users:
        config_path = os.path.join(config_root, user, 'cpc_config.json')
        with open(config_path, 'r') as f:
            CONFIG = json.load(f)

        CONFIG['data_dir'] = f'./dataset/{user}'
        CONFIG['output_dir'] = f'./pretrained_cpc_models/{user}'
        if not torch.cuda.is_available():
            CONFIG['device'] = 'cpu'

        norm_stat = compute_and_save_normalization_stats(
            CONFIG['data_dir'],
            CONFIG['output_dir']
        )
        CONFIG['normalization_stat'] = norm_stat

        model = train_cpc(config=CONFIG)
