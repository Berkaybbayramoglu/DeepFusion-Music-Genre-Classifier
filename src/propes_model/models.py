from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MelCNN(nn.Module):
    def __init__(self, output_dim: int = 256, pretrained: bool = True) -> None:
        super(MelCNN, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Sequential(nn.Linear(2048, output_dim), nn.ReLU(), nn.Dropout(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class AuxiliaryFeatureMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size1: int, hidden_size2: int, dropout_rate: float = 0.5) -> None:
        super(AuxiliaryFeatureMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MultiModalNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(MultiModalNet, self).__init__()
        self.mel_orig_branch = MelCNN(pretrained=True, output_dim=256)
        self.mel_harm_branch = MelCNN(pretrained=True, output_dim=256)
        self.mel_perc_branch = MelCNN(pretrained=True, output_dim=256)

        self.mfcc_branch = AuxiliaryFeatureMLP(input_size=20, hidden_size1=128, hidden_size2=32, dropout_rate=0.5)
        self.chroma_branch = AuxiliaryFeatureMLP(input_size=12, hidden_size1=64, hidden_size2=16, dropout_rate=0.5)
        self.tempogram_branch = AuxiliaryFeatureMLP(input_size=384, hidden_size1=128, hidden_size2=32, dropout_rate=0.5)

        self.mel_output_dim = 256
        self.mfcc_output_dim = 32
        self.chroma_output_dim = 16
        self.tempogram_output_dim = 32

        self.num_modalities = 6

        self.total_feature_dim = (
            self.mel_output_dim * 3
            + self.mfcc_output_dim
            + self.chroma_output_dim
            + self.tempogram_output_dim
        )

        self.attention_weights_layer = nn.Linear(self.total_feature_dim, self.num_modalities)

        self.fusion_fc = nn.Sequential(
            nn.Linear(self.total_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.6),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        mel_orig_output = self.mel_orig_branch(x['mel_orig'])
        mel_harm_output = self.mel_harm_branch(x['mel_harm'])
        mel_perc_output = self.mel_perc_branch(x['mel_perc'])

        mfcc_output = self.mfcc_branch(x['mfcc'])
        chroma_output = self.chroma_branch(x['chroma'])
        tempogram_output = self.tempogram_branch(x['tempogram'])

        combined_features_raw = torch.cat(
            (mel_orig_output, mel_harm_output, mel_perc_output, mfcc_output, chroma_output, tempogram_output), dim=1
        )

        attention_logits = self.attention_weights_layer(combined_features_raw)
        attention_weights = F.softmax(attention_logits, dim=1)

        weighted_mel_orig_output = mel_orig_output * attention_weights[:, 0].unsqueeze(1)
        weighted_mel_harm_output = mel_harm_output * attention_weights[:, 1].unsqueeze(1)
        weighted_mel_perc_output = mel_perc_output * attention_weights[:, 2].unsqueeze(1)
        weighted_mfcc_output = mfcc_output * attention_weights[:, 3].unsqueeze(1)
        weighted_chroma_output = chroma_output * attention_weights[:, 4].unsqueeze(1)
        weighted_tempogram_output = tempogram_output * attention_weights[:, 5].unsqueeze(1)

        combined_features_weighted = torch.cat(
            (
                weighted_mel_orig_output,
                weighted_mel_harm_output,
                weighted_mel_perc_output,
                weighted_mfcc_output,
                weighted_chroma_output,
                weighted_tempogram_output,
            ),
            dim=1,
        )

        fusion_output = self.fusion_fc(combined_features_weighted)
        output = self.classifier(fusion_output)
        return output
