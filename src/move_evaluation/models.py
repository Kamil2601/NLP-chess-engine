import torch.nn as nn

class SentimateNet(nn.Module):
    def __init__(self, dropout = 0.25) -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(26, 26, 5, padding='same'),
            nn.Dropout(dropout),
            nn.Conv2d(26, 26, 3, padding='same'),
            nn.Flatten()
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(8*8*26, 500),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(500, 200),
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
    def __init__(self, dropout = 0.25) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(26, 26, 5, padding='same'),
            nn.Dropout(dropout),
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
            nn.Linear(200, 1)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        conv1_2_out = self.conv2(conv1_out)

        fc_input = (conv1_out + conv2_out + conv1_2_out)/3

        x = self.fc_layers(x)
        return x