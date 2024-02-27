from turtle import forward
import torch.nn as nn
import torch.nn.functional as F


class GoodProbability(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        x = F.softmax(x, dim=-1)
        x = x[:, 1]
        return x
        

class SentimateNet(nn.Module):
    def __init__(self, input_channels = 26, dropout = 0.25, output_size = 2, skip_connection = False) -> None:
        super().__init__()
        self.skip_connection = skip_connection

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 5, padding='same'),
            nn.ELU(),
            nn.Conv2d(input_channels, input_channels, 3, padding='same'),
        )

        self.fc_layers = nn.Sequential(
            nn.ELU(),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(8*8*input_channels, 500),
            nn.ELU(),
            nn.Linear(500, 200),
            nn.ELU(),
            nn.Linear(200, output_size)
        )

    def forward(self, x):
        out = self.conv_layers(x)
        
        if self.skip_connection:
            out = (out + x)

        out = self.fc_layers(out)
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, input_channels = 26) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, 3, padding='same')
        self.conv2 = nn.Conv2d(input_channels, input_channels, 3, padding='same')
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(input_channels)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm(x)
        x += identity
        x = self.relu(x)
        return x
    
class ValueHead(nn.Module):
    def __init__(self, input_channels = 26, output_size = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(8*8*input_channels),
            nn.ReLU(),
            nn.Linear(8*8*26, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.layers(x)
    
class AlphaZeroOnlyValueHead(nn.Module):
    def __init__(self, input_channels = 26, num_blocks = 2, output_size = 1) -> None:
        super().__init__()
        self.residual_blocks = nn.Sequential(*[ResidualBlock(input_channels = input_channels) for _ in range(num_blocks)])
        self.value_head = ValueHead(input_channels=input_channels, output_size = output_size)

    def forward(self, x):
        x = self.residual_blocks(x)
        x = self.value_head(x)
        return x

class SentimateNetWithBatchNorm(nn.Module):
    def __init__(self, dropout = 0.25) -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(26, 26, 5, padding='same'),
            nn.BatchNorm2d(26),
            nn.Dropout(dropout),
            nn.Conv2d(26, 26, 3, padding='same'),
            nn.BatchNorm2d(26),
            nn.Flatten()
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(8*8*26, 500),
            nn.BatchNorm1d(500),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(500, 200),
            nn.LazyBatchNorm1d(200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(200, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class SentimateNetSmaller(nn.Module):
    def __init__(self, dropout = 0.25) -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(26, 13, 5, padding='same'),
            nn.Dropout(dropout),
            nn.Conv2d(13, 26, 3, padding='same'),
            nn.Flatten()
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(8*8*26, 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class SentimateNetSkipLayer(nn.Module):
    def __init__(self, dropout = 0.25, output_size = 2) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(26, 26, 5, padding='same'),
            # nn.Dropout(dropout),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(26, 26, 3, padding='same'),
            nn.Dropout(dropout),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*8*26, 500),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(500, 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(200, output_size)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        conv1_2_out = self.conv2(conv1_out)

        fc_input = (conv1_out + conv2_out + conv1_2_out)/3

        x = self.fc_layers(x)
        return x