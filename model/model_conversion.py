import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import ai_edge_torch
import numpy
import torchvision
import os
import numpy as np
import json


###########################################
# MODEL DEFINITION
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
    def __init__(self, input_dim=6, d_model=128, nhead=4, num_layers=3, num_classes=6,
                 dropout=0.1, max_seq_length=128,
                 temporal_kernel_size=8, temporal_stride=1):
        super().__init__()

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

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch_size, orig_seq_len, feature_dim = x.shape

        x = x.transpose(1, 2)
        x = self.temporal_tokenizer(x)
        x = x.transpose(1, 2)

        x = self.positional_encoding(x)

        x = self.transformer_encoder(x)

        x = torch.mean(x, dim=1)
        return self.classifier(x)


###########################################
# METADATA EXPORT
###########################################

def create_metadata_file(class_mapping_id_name, norm_stats, output_path="model_metadata.json", config={}):
    """Create metadata file with class mapping and normalization stats for Android"""
    import json

    class_mapping = {}
    for k in class_mapping_id_name.keys():
        class_mapping[class_mapping_id_name[k]] = k

    metadata = {
        "class_mapping": class_mapping,
        "normalization": {
            "mean": norm_stats["mean"].tolist() if norm_stats and "mean" in norm_stats else [],
            "std": norm_stats["std"].tolist() if norm_stats and "std" in norm_stats else [],
            "feature_names": norm_stats.get("feature_names",
                                            []).tolist() if norm_stats and "feature_names" in norm_stats else []
        },
        "input_shape": [1, config.get("window_size", 120), config.get("input_dim", 6)],
        "model_info": {
            "name": "IMU Transformer",
            "task": "classification"
        }
    }

    try:
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata file saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error creating metadata file: {e}")
        return False



###########################################
# MODEL CONVERSION
###########################################

model_path = "./output/P04/pretraining_comparison/contrastive/best_supervised_model.pth"
class_map_file = os.path.join(
    os.path.dirname(
        model_path
    ),
    "class_mapping.json"
)

with open(class_map_file, 'r') as f:
    class_map = json.load(f)

total_class = len(class_map)

model_weight = torch.load(model_path)['model_state_dict']

model = IMUTransformer(
    input_dim=6,
    d_model=128,
    nhead=4,
    num_layers=3,
    num_classes=total_class,
    dropout=0.2,
    max_seq_length=120,
    temporal_kernel_size=8,
    temporal_stride=1
)

model.load_state_dict(model_weight)

model.eval()

csv = "./dataset/P10/test/rest/circle_test_1_resting_1.csv"
norm_file = "./output/P10/normalization_stats.npz"
data = np.array(pd.read_csv(csv).iloc[:120])[:, 1:7]

norm_utils = np.load(norm_file)

json_config_path = os.path.join(os.path.dirname(model_path), "training_config.json")
with open(json_config_path, 'r') as f:
    config = json.load(f)

create_metadata_file(class_map, norm_utils, './edge_models/model_metadata.json', config)

data = (data - norm_utils['mean'])/norm_utils['std']

data = np.expand_dims(data, axis=0).astype(np.float32)

with torch.no_grad():
    sample_input = (torch.from_numpy(data),)

    edge_model = ai_edge_torch.convert(model.eval(), sample_input)

    print(sample_input[0].shape)

    output = edge_model(*sample_input)

print(output)

output = model(torch.from_numpy(data))

print(output)

edge_model.export('./edge_models/imu_transformer.tflite')
