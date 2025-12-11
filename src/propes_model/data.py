from typing import Any, Dict, List, Tuple
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter


class MultiModalDataset(Dataset):
    def __init__(
        self,
        mel_orig_data: np.ndarray,
        mel_harm_data: np.ndarray,
        mel_perc_data: np.ndarray,
        mfcc_data: np.ndarray,
        chroma_data: np.ndarray,
        tempogram_data: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.mel_orig_data = torch.tensor(mel_orig_data, dtype=torch.float32)
        self.mel_harm_data = torch.tensor(mel_harm_data, dtype=torch.float32)
        self.mel_perc_data = torch.tensor(mel_perc_data, dtype=torch.float32)

        self.mfcc_data = torch.tensor(mfcc_data, dtype=torch.float32)
        self.chroma_data = torch.tensor(chroma_data, dtype=torch.float32)
        self.tempogram_data = torch.tensor(tempogram_data, dtype=torch.float32)

        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            'mel_orig': self.mel_orig_data[idx],
            'mel_harm': self.mel_harm_data[idx],
            'mel_perc': self.mel_perc_data[idx],
            'mfcc': self.mfcc_data[idx],
            'chroma': self.chroma_data[idx],
            'tempogram': self.tempogram_data[idx],
        }, self.labels[idx]


def load_features(file_path: str) -> Tuple[List[Dict[str, Any]], List[int], List[str]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Features file not found: '{file_path}'")
    with open(file_path, 'rb') as f:
        all_clips_raw_data = pickle.load(f)
    all_features = [item['features'] for item in all_clips_raw_data]
    all_labels = [item['label'] for item in all_clips_raw_data]
    original_files = [item.get('original_file', '') for item in all_clips_raw_data]
    return all_features, all_labels, original_files


def preprocess_all_features(all_features: List[Dict[str, Any]], all_labels: List[int]):
    mel_orig_specs = np.array([f['mel_spec_orig'] for f in all_features])
    mel_harm_specs = np.array([f['mel_spec_harm'] for f in all_features])
    mel_perc_specs = np.array([f['mel_spec_perc'] for f in all_features])
    mfccs = np.array([f['mfcc'] for f in all_features])
    chromas = np.array([f['chroma'] for f in all_features])
    tempograms = np.array([f['tempogram'] for f in all_features])

    labels = np.array(all_labels)

    if mel_orig_specs.ndim == 3:
        mel_orig_specs = np.expand_dims(mel_orig_specs, axis=1)
    if mel_harm_specs.ndim == 3:
        mel_harm_specs = np.expand_dims(mel_harm_specs, axis=1)
    if mel_perc_specs.ndim == 3:
        mel_perc_specs = np.expand_dims(mel_perc_specs, axis=1)

    return mel_orig_specs, mel_harm_specs, mel_perc_specs, mfccs, chromas, tempograms, labels


def create_data_loaders(
    all_mel_orig: np.ndarray,
    all_mel_harm: np.ndarray,
    all_mel_perc: np.ndarray,
    all_mfcc: np.ndarray,
    all_chroma: np.ndarray,
    all_tempogram: np.ndarray,
    all_labels: np.ndarray,
    batch_size: int,
    validation_split_ratio: float,
    random_seed: int,
):
    num_samples = len(all_labels)
    indices = np.arange(num_samples)

    train_indices, val_indices = train_test_split(
        indices, test_size=validation_split_ratio, stratify=all_labels, random_state=random_seed
    )

    X_train_mel_orig = all_mel_orig[train_indices]
    X_train_mel_harm = all_mel_harm[train_indices]
    X_train_mel_perc = all_mel_perc[train_indices]
    X_train_mfcc = all_mfcc[train_indices]
    X_train_chroma = all_chroma[train_indices]
    X_train_tempogram = all_tempogram[train_indices]
    y_train = all_labels[train_indices]

    X_val_mel_orig = all_mel_orig[val_indices]
    X_val_mel_harm = all_mel_harm[val_indices]
    X_val_mel_perc = all_mel_perc[val_indices]
    X_val_mfcc = all_mfcc[val_indices]
    X_val_chroma = all_chroma[val_indices]
    X_val_tempogram = all_tempogram[val_indices]
    y_val = all_labels[val_indices]

    features_to_scale = {
        'mfcc': (X_train_mfcc, X_val_mfcc),
        'chroma': (X_train_chroma, X_val_chroma),
        'tempogram': (X_train_tempogram, X_val_tempogram),
    }

    scalers = {}
    scaled_train_features = {}
    scaled_val_features = {}

    for name, (train_data, val_data) in features_to_scale.items():
        scaler = StandardScaler()
        scaled_train_features[name] = scaler.fit_transform(train_data)
        scaled_val_features[name] = scaler.transform(val_data)
        scalers[name] = scaler

    train_dataset = MultiModalDataset(
        X_train_mel_orig,
        X_train_mel_harm,
        X_train_mel_perc,
        scaled_train_features['mfcc'],
        scaled_train_features['chroma'],
        scaled_train_features['tempogram'],
        y_train,
    )
    val_dataset = MultiModalDataset(
        X_val_mel_orig,
        X_val_mel_harm,
        X_val_mel_perc,
        scaled_val_features['mfcc'],
        scaled_val_features['chroma'],
        scaled_val_features['tempogram'],
        y_val,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    return train_loader, val_loader, y_train
