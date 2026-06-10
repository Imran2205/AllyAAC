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
    Preprocess raw IMU data for contrastive learning

    Parameters:
    - data: DataFrame with columns for acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, etc.
    - config: Dictionary containing configuration parameters

    Returns:
    - windowed_data: Array of shape [n_windows, window_size, n_features]
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


class ContrastiveIMUDataset(Dataset):
    def __init__(self, data_list, config=None):
        """
        Initialize the dataset

        Parameters:
        - data_list: List of preprocessed IMU data arrays from different sessions
        - config: Configuration dictionary
        """
        if config is None:
            config = {}

        all_windows = []
        for session_data in data_list:
            all_windows.append(session_data)

        self.data = np.concatenate(all_windows, axis=0)
        self.augmenter = IMUDataAugmenter(config)

        # Contrastive learning parameters
        self.negative_mining_strategy = config.get('negative_mining_strategy', 'random')
        self.num_negatives = config.get('num_negatives', 1)
        self.temporal_distance = config.get('temporal_distance', 10)

        print(f"Created contrastive dataset with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return an anchor sample, its positive pair (augmented version),
        and one or more negative samples (different windows)
        """
        anchor = self.data[idx]
        anchor_tensor = torch.tensor(anchor, dtype=torch.float32)

        positive = self.augmenter(anchor_tensor.unsqueeze(0)).squeeze(0)

        if self.negative_mining_strategy == 'random':
            negative_indices = []
            for _ in range(self.num_negatives):
                negative_idx = random.randint(0, len(self.data) - 1)
                while negative_idx == idx:
                    negative_idx = random.randint(0, len(self.data) - 1)
                negative_indices.append(negative_idx)

        elif self.negative_mining_strategy == 'temporal':
            negative_indices = []
            for _ in range(self.num_negatives):
                candidate_indices = list(range(max(0, idx - self.temporal_distance))) + \
                                    list(range(idx + self.temporal_distance, len(self.data)))
                if not candidate_indices:  # If no valid indices found
                    candidate_indices = list(range(len(self.data)))
                    candidate_indices.remove(idx)
                negative_idx = random.choice(candidate_indices)
                negative_indices.append(negative_idx)

        elif self.negative_mining_strategy == 'hard':
            negative_indices = []
            for _ in range(self.num_negatives):
                min_dist = 3
                max_dist = 7

                low_range = list(range(max(0, idx - max_dist), max(0, idx - min_dist)))
                high_range = list(range(min(len(self.data), idx + min_dist),
                                        min(len(self.data), idx + max_dist)))

                candidate_indices = low_range + high_range
                if not candidate_indices:
                    candidate_indices = list(range(len(self.data)))
                    candidate_indices.remove(idx)
                negative_idx = random.choice(candidate_indices)
                negative_indices.append(negative_idx)

        else:
            negative_indices = [random.randint(0, len(self.data) - 1) for _ in range(self.num_negatives)]

        negatives = [torch.tensor(self.data[neg_idx], dtype=torch.float32)
                     for neg_idx in negative_indices]

        negatives = torch.stack(negatives)

        return anchor_tensor, positive, negatives


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


class IMUContrastiveModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, projection_dim=64,
                 dropout=0.1, max_seq_length=128, temporal_kernel_size=8, temporal_stride=1):
        super().__init__()

        self.input_dim = input_dim
        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_stride = temporal_stride
        self.projection_dim = projection_dim

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

        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim)
        )

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape

        x = x.transpose(1, 2)
        x = self.temporal_tokenizer(x)
        x = x.transpose(1, 2)

        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)
        embeddings = self.projection_head(x)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


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

        self.use_permute_segments = config.get('permute_segments', False)
        self.permutation_segments = config.get('permutation_segments', 3)
        self.use_mask_segments = config.get('mask_segments', False)
        self.mask_ratio = config.get('mask_ratio', 0.15)

        if config.get('contrastive_augmentation_strength', 'normal') == 'strong':
            self.jitter_scale *= 1.5
            self.time_warp_scale *= 1.5
            self.rotation_angle *= 1.5
            self.magnitude_scale *= 1.5

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

    def permute_segments_func(self, x):
        batch_size, seq_len, features = x.shape
        if seq_len < self.permutation_segments * 2:
            return x

        segment_size = seq_len // self.permutation_segments

        permuted_x = x.clone()
        for i in range(batch_size):
            segment_indices = torch.randperm(self.permutation_segments)

            for j, idx in enumerate(segment_indices):
                start_src = idx * segment_size
                end_src = min((idx + 1) * segment_size, seq_len)

                start_dst = j * segment_size
                end_dst = min((j + 1) * segment_size, seq_len)

                seg_len = min(end_src - start_src, end_dst - start_dst)
                permuted_x[i, start_dst:start_dst + seg_len] = x[i, start_src:start_src + seg_len]

        return permuted_x

    def mask_random_segments_func(self, x):
        batch_size, seq_len, features = x.shape
        num_masks = int(seq_len * self.mask_ratio)
        if num_masks == 0:
            return x

        masked_x = x.clone()
        for i in range(batch_size):
            if random.random() < 0.5:
                segment_size = random.randint(3, max(4, num_masks // 2))
                num_segments = num_masks // segment_size

                for _ in range(num_segments):
                    start_idx = random.randint(0, seq_len - segment_size)
                    masked_x[i, start_idx:start_idx + segment_size] = 0.0
            else:
                mask_indices = torch.randperm(seq_len)[:num_masks]
                masked_x[i, mask_indices] = 0.0

        return masked_x

    def __call__(self, x):
        augmentations = [
            self.jitter,
            self.scale_magnitude,
            self.time_warp,
            self.rotate
        ]

        if self.use_permute_segments:
            augmentations.append(self.permute_segments_func)
        if self.use_mask_segments:
            augmentations.append(self.mask_random_segments_func)

        num_augmentations = torch.randint(2, min(len(augmentations) + 1, 4), (1,)).item()
        chosen_indices = torch.randperm(len(augmentations))[:num_augmentations].tolist()

        augmented_x = x.clone()
        for aug_idx in chosen_indices:
            augmented_x = augmentations[aug_idx](augmented_x)

        return augmented_x


###########################################
# 4. CONTRASTIVE LEARNING LOSS
###########################################

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, return_logits=False):
        super().__init__()
        self.temperature = temperature
        self.return_logits = return_logits
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_function = nn.CosineSimilarity(dim=2)

    def forward(self, anchor_embeddings, positive_embeddings, negative_embeddings=None, return_logits=None):
        """
        Calculate loss for batch of anchor, positive and optional negative embeddings

        Parameters:
        - anchor_embeddings: Tensor of shape [batch_size, embedding_dim]
        - positive_embeddings: Tensor of shape [batch_size, embedding_dim]
        - negative_embeddings: Optional tensor of shape [batch_size, num_negatives, embedding_dim]
          If None, other samples in the batch will be used as negatives
        - return_logits: Whether to return logits (overrides the value set in __init__)

        Returns:
        - Loss value or (loss, logits) tuple if return_logits is True
        """
        # Allow return_logits to be overridden in the call
        return_logits = self.return_logits if return_logits is None else return_logits

        device = anchor_embeddings.device
        batch_size = anchor_embeddings.shape[0]

        if negative_embeddings is None:
            features = torch.cat([anchor_embeddings, positive_embeddings], dim=0)
            similarity_matrix = torch.matmul(features, features.T) / self.temperature
            mask = torch.eye(batch_size * 2, dtype=torch.bool, device=device)
            similarity_matrix = similarity_matrix[~mask].view(batch_size * 2, batch_size * 2 - 1)
            positive_mask = torch.zeros((batch_size * 2, batch_size * 2 - 1), dtype=torch.bool, device=device)
            for i in range(batch_size):
                positive_mask[i, batch_size - 1 + i] = True
                positive_mask[i + batch_size, i] = True

            labels = torch.zeros(batch_size * 2, dtype=torch.long, device=device)
            for i in range(batch_size * 2):
                labels[i] = positive_mask[i].nonzero(as_tuple=True)[0]

            loss = self.criterion(similarity_matrix, labels)
            loss = loss / (2 * batch_size)
            if self.return_logits:
                return loss, similarity_matrix

            return loss
        else:
            num_negatives = negative_embeddings.shape[1]
            anchors_expanded = anchor_embeddings.unsqueeze(1)
            positives_expanded = positive_embeddings.unsqueeze(1)
            positive_scores = self.similarity_function(anchors_expanded, positives_expanded) / self.temperature
            negative_scores = self.similarity_function(
                anchors_expanded.expand(-1, num_negatives, -1),
                negative_embeddings
            ) / self.temperature
            logits = torch.cat([positive_scores, negative_scores], dim=1)
            labels = torch.zeros(batch_size, dtype=torch.long, device=device)
            loss = self.criterion(logits, labels)
            loss = loss / batch_size
            if return_logits:
                return loss, logits

            return loss


###########################################
# 5. CONTRASTIVE TRAINING
###########################################

class ContrastiveTrainer:
    def __init__(self, model, config=None, device=None):
        if config is None:
            config = {}

        if device is None:
            device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model
        self.device = device
        self.model.to(device)

        lr = config.get('learning_rate', 1e-4)
        weight_decay = config.get('weight_decay', 1e-5)
        self.temperature = config.get('temperature', 0.5)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        factor = config.get('lr_factor', 0.5)
        patience = config.get('patience', 5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=factor, patience=patience
        )
        self.criterion = NTXentLoss(temperature=self.temperature)

    def train_epoch(self, dataloader, config=None):
        if config is None:
            config = {}

        self.model.train()
        total_loss = 0
        num_batches = 0
        for batch_idx, (anchors, positives, negatives) in enumerate(dataloader):
            anchors = anchors.to(self.device)
            positives = positives.to(self.device)
            negatives = negatives.to(self.device)

            self.optimizer.zero_grad()

            anchor_embeddings = self.model(anchors)
            positive_embeddings = self.model(positives)
            batch_size, num_negatives = negatives.shape[0], negatives.shape[1]
            seq_len, features = negatives.shape[2], negatives.shape[3]
            negatives_reshaped = negatives.view(batch_size * num_negatives, seq_len, features)
            negative_embeddings = self.model(negatives_reshaped)
            embedding_dim = negative_embeddings.shape[1]
            negative_embeddings = negative_embeddings.view(batch_size, num_negatives, embedding_dim)
            loss = self.criterion(anchor_embeddings, positive_embeddings, negative_embeddings)

            loss.backward()

            if config.get('gradient_clipping', True):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.get('clip_value', 1.0))

            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % config.get('print_freq', 10) == 0:
                print(f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.6f}")

        avg_loss = total_loss / num_batches
        print(f"Training Loss: {avg_loss:.6f}")

        return avg_loss

    def evaluate(self, dataloader, config=None):
        if config is None:
            config = {}

        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_similarities = []
        positive_similarities = []
        negative_similarities = []

        with torch.no_grad():
            for anchors, positives, negatives in dataloader:
                anchors = anchors.to(self.device)
                positives = positives.to(self.device)
                negatives = negatives.to(self.device)
                anchor_embeddings = self.model(anchors)
                positive_embeddings = self.model(positives)
                batch_size, num_negatives = negatives.shape[0], negatives.shape[1]
                seq_len, features = negatives.shape[2], negatives.shape[3]
                negatives_reshaped = negatives.view(batch_size * num_negatives, seq_len, features)
                negative_embeddings = self.model(negatives_reshaped)
                embedding_dim = negative_embeddings.shape[1]
                negative_embeddings = negative_embeddings.view(batch_size, num_negatives, embedding_dim)
                loss, logits = self.criterion(anchor_embeddings, positive_embeddings,
                                             negative_embeddings, return_logits=True)

                total_loss += loss.item()
                num_batches += 1

                pos_sim = logits[:, 0].cpu().numpy()
                neg_sim = logits[:, 1:].cpu().numpy()
                positive_similarities.extend(pos_sim)
                negative_similarities.extend(neg_sim.flatten())
                all_similarities.extend(logits.cpu().numpy().flatten())

        avg_loss = total_loss / num_batches
        avg_pos_sim = np.mean(positive_similarities)
        avg_neg_sim = np.mean(negative_similarities)
        alignment = -np.log(np.mean(np.exp(np.array(positive_similarities))))
        uniformity = np.log(np.mean(np.exp(-2 * np.array(all_similarities))))
        print(f"Validation Loss: {avg_loss:.6f}")
        print(f"Avg. Positive Similarity: {avg_pos_sim:.4f}, Avg. Negative Similarity: {avg_neg_sim:.4f}")
        print(f"Alignment: {alignment:.4f}, Uniformity: {uniformity:.4f}")
        metrics = {
            'loss': avg_loss,
            'positive_similarity': avg_pos_sim,
            'negative_similarity': avg_neg_sim,
            'alignment': alignment,
            'uniformity': uniformity
        }

        return metrics

    def visualize_embeddings(self, dataloader, num_samples=500, plot_file='embeddings.png'):
        """
        Visualize the learned embeddings using t-SNE or PCA

        Parameters:
        - dataloader: DataLoader to sample from
        - num_samples: Maximum number of samples to plot
        - plot_file: Path to save the plot
        """
        try:
            import umap
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt

            embeddings = []
            with torch.no_grad():
                count = 0
                for anchors, _, _ in dataloader:
                    anchors = anchors.to(self.device)
                    batch_embeddings = self.model(anchors).cpu().numpy()
                    embeddings.append(batch_embeddings)

                    count += anchors.size(0)
                    if count >= num_samples:
                        break

            embeddings = np.vstack(embeddings)[:num_samples]
            plt.figure(figsize=(18, 6))
            try:
                plt.subplot(1, 3, 1)
                n_neighbors = min(15, max(2, embeddings.shape[0] // 5))
                reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42)
                embedding_umap = reducer.fit_transform(embeddings)
                plt.scatter(embedding_umap[:, 0], embedding_umap[:, 1], s=5, alpha=0.5)
                plt.title('UMAP Projection')
                plt.xlabel('UMAP 1')
                plt.ylabel('UMAP 2')
            except:
                print("UMAP projection failed, skipping")

            plt.subplot(1, 3, 2)
            perplexity = min(30, max(5, embeddings.shape[0] // 5))
            embedding_tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(embeddings)
            plt.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], s=5, alpha=0.5)
            plt.title('t-SNE Projection')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')

            plt.subplot(1, 3, 3)
            embedding_pca = PCA(n_components=2).fit_transform(embeddings)
            plt.scatter(embedding_pca[:, 0], embedding_pca[:, 1], s=5, alpha=0.5)
            plt.title('PCA Projection')
            plt.xlabel('PC 1')
            plt.ylabel('PC 2')

            plt.tight_layout()
            plt.savefig(plot_file, dpi=300)
            plt.close()

            print(f"Embedding visualization saved to {plot_file}")
        except Exception as e:
            print(f"Visualization failed: {e}")


###########################################
# 6. TRAINING
###########################################

def train_contrastive_learning(config=None):
    """
    Run the contrastive learning training pipeline

    Parameters:
    - config: Dictionary containing configuration parameters

    Returns:
    - Trained model
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
    config['learning_rate'] = config.get('learning_rate', 1e-4)
    config['lr_factor'] = config.get('lr_factor', 0.5)
    config['patience'] = config.get('patience', 5)
    config['weight_decay'] = config.get('weight_decay', 1e-5)
    config['sampling_rate'] = config.get('sampling_rate', 50)
    config['val_split'] = config.get('val_split', 0.1)
    config['jitter_scale'] = config.get('jitter_scale', 0.1)
    config['time_warp_scale'] = config.get('time_warp_scale', 0.2)
    config['rotation_angle'] = config.get('rotation_angle', 10)
    config['magnitude_scale'] = config.get('magnitude_scale', 0.1)
    config['random_seed'] = config.get('random_seed', 42)
    config['temperature'] = config.get('temperature', 0.5)
    config['projection_dim'] = config.get('projection_dim', 64)
    config['num_negatives'] = config.get('num_negatives', 3)
    config['negative_mining_strategy'] = config.get('negative_mining_strategy', 'random')
    config['temporal_distance'] = config.get('temporal_distance', 10)
    config['contrastive_augmentation_strength'] = config.get('contrastive_augmentation_strength', 'strong')
    config['permute_segments'] = config.get('permute_segments', True)
    config['permutation_segments'] = config.get('permutation_segments', 3)
    config['mask_segments'] = config.get('mask_segments', True)
    config['mask_ratio'] = config.get('mask_ratio', 0.15)
    config['early_stopping_patience'] = config.get('early_stopping_patience', 10)
    config['gradient_clipping'] = config.get('gradient_clipping', True)
    config['clip_value'] = config.get('clip_value', 1.0)
    config['temporal_kernel_size'] = config.get('temporal_kernel_size', 8)
    config['temporal_stride'] = config.get('temporal_stride', 1)

    set_seed(config.get('random_seed', 42))

    device = config['device']
    print(f"Using device: {device}")
    os.makedirs(config['output_dir'], exist_ok=True)
    print("Loading and preprocessing data...")
    train_data = load_all_sessions(config, split='train')
    val_data = load_all_sessions(config, split='test')

    # train_data = load_all_sessions(config, split=None)
    # val_data = load_all_sessions(config, split=None)
    # #
    # val_data = train_data[-len(train_data) // 5:]
    # train_data = train_data[:-len(train_data) // 5]

    if not train_data:
        print("No data found. Exiting.")
        return

    input_dim = train_data[0].shape[-1]
    print(f"Input dimension: {input_dim}")
    config['input_dim'] = input_dim
    train_dataset = ContrastiveIMUDataset(train_data)
    train_dataset_size = len(train_dataset)

    val_dataset = ContrastiveIMUDataset(val_data)
    val_dataset_size = len(val_dataset)
    dataset_size = train_dataset_size + val_dataset_size
    print(f"Dataset size: {dataset_size}")

    # Split into training and validation sets
    # val_split = config.get('val_split', 0.1)
    # train_size = int((1.0 - val_split) * dataset_size)
    # val_size = dataset_size - train_size
    #
    # generator = torch.Generator().manual_seed(config['random_seed'])
    # train_dataset, val_dataset = torch.utils.data.random_split(
    #     dataset,
    #     [train_size, val_size],
    #     generator=generator
    # )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False)
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False)
    )

    print(f"Training dataset size: {len(train_dataloader)}, Val dataset size: {len(val_dataloader)}")

    print("Creating contrastive learning model...")
    model = IMUContrastiveModel(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        projection_dim=config['projection_dim'],
        dropout=config['dropout'],
        max_seq_length=config['window_size'],
        temporal_kernel_size=config['temporal_kernel_size'],
        temporal_stride=config['temporal_stride']
    )

    trainer = ContrastiveTrainer(model, config=config)
    print("Starting training...")
    best_val_loss = float('inf')
    best_alignment = float('inf')
    patience_counter = 0
    early_stopping_patience = config.get('early_stopping_patience', 10)

    train_losses = []
    val_losses = []
    val_metrics = []
    learning_rates = []

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        current_lr = trainer.optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        train_loss = trainer.train_epoch(train_dataloader, config=config)
        train_losses.append(train_loss)
        metrics = trainer.evaluate(val_dataloader, config=config)
        val_loss = metrics['loss']
        val_losses.append(val_loss)
        val_metrics.append(metrics)
        trainer.scheduler.step(val_loss)
        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = True

            model_path = os.path.join(config['output_dir'], 'best_contrastive_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': metrics,
                'config': config,
            }, model_path)
            print(f"  Saved best model to {model_path} (Val Loss: {val_loss:.6f})")

        if metrics['alignment'] < best_alignment:
            best_alignment = metrics['alignment']
            improved = True

            model_path = os.path.join(config['output_dir'], 'best_alignment_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': metrics,
                'config': config,
            }, model_path)
            print(f"  Saved model with best alignment to {model_path} (Alignment: {best_alignment:.6f})")

        if improved:
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping after {patience_counter} epochs without improvement")
                break

        if (epoch + 1) % 5 == 0 or epoch == config['num_epochs'] - 1:
            viz_file = os.path.join(config['output_dir'], f'embeddings_epoch_{epoch+1}.png')
            trainer.visualize_embeddings(val_dataloader, plot_file=viz_file)

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    alignment_values = [metrics['alignment'] for metrics in val_metrics]
    uniformity_values = [metrics['uniformity'] for metrics in val_metrics]
    plt.plot(alignment_values, label='Alignment')
    plt.plot(uniformity_values, label='Uniformity')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Alignment and Uniformity')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    pos_sim_values = [metrics['positive_similarity'] for metrics in val_metrics]
    neg_sim_values = [metrics['negative_similarity'] for metrics in val_metrics]
    plt.plot(pos_sim_values, label='Positive Similarity')
    plt.plot(neg_sim_values, label='Negative Similarity')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.title('Positive and Negative Similarities')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate')
    plt.grid(True)
    if len(learning_rates) > 0:
        plt.yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], 'contrastive_training_curves.png'))
    plt.close()

    final_model_path = os.path.join(config['output_dir'], 'final_contrastive_model.pth')
    torch.save({
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'val_loss': val_loss,
        'val_metrics': metrics,
        'best_val_loss': best_val_loss,
        'config': config,
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")

    trainer.visualize_embeddings(
        val_dataloader,
        plot_file=os.path.join(config['output_dir'], 'final_embeddings.png')
    )
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best alignment: {best_alignment:.6f}")

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
    users = ['P03', 'P04', 'P05', 'P06']
    for user in users:
        CONFIG = {
            'data_dir': f'./dataset/{user}',  # Directory containing CSV files
            'output_dir': f'./pretrained_contrastive_models/{user}',  # Directory to save trained models
            'window_size': 120,  # Number of samples in each window
            'overlap': 0.7,  # Fraction of overlap between consecutive windows
            'batch_size': 32,  # Batch size for training
            'num_epochs': 200,  # Number of training epochs
            'd_model': 128,  # Model dimension
            'nhead': 4,  # Number of attention heads
            'num_layers': 3,  # Number of transformer layers
            'dropout': 0.2,  # Dropout rate
            'device': 'cuda:3' if torch.cuda.is_available() else 'cpu',  # Device for training
            'learning_rate': 5e-4,  # Learning rate for optimizer
            'weight_decay': 1e-5,  # Weight decay for regularization
            'lr_factor': 0.7,  # Factor by which to reduce learning rate on plateau
            'patience': 5,  # Number of epochs to wait before reducing learning rate
            'sampling_rate': 50,  # IMU data sampling rate in Hz
            'val_split': 0.2,  # Validation split ratio
            'jitter_scale': 0.1,  # Scale of jitter noise
            'time_warp_scale': 0.2,  # Scale of time warping
            'rotation_angle': 10,  # Maximum rotation angle in degrees
            'magnitude_scale': 0.1,  # Scale for magnitude scaling
            'random_seed': 42,
            'temperature': 0.07,  # Temperature parameter for NT-Xent loss
            'projection_dim': 64,  # Dimension of projection head output
            'num_negatives': 5,  # Number of negative samples per positive pair
            'negative_mining_strategy': 'temporal',  # Strategy for negative mining: 'random', 'temporal', or 'hard'
            'temporal_distance': 50,  # Minimum temporal distance for negative samples
            'contrastive_augmentation_strength': 'strong',  # Augmentation strength: 'normal' or 'strong'
            'permute_segments': True,  # Whether to use permutation augmentation
            'mask_segments': True,  # Whether to use masking augmentation
            'early_stopping_patience': 50,  # Patience for early stopping
        }

        norm_stat = compute_and_save_normalization_stats(
            CONFIG['data_dir'],
            CONFIG['output_dir']
        )

        CONFIG['normalization_stat'] = norm_stat

        model = train_contrastive_learning(config=CONFIG)