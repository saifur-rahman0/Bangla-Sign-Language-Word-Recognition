import torch
import torch.nn as nn


class CNNBiLSTMAttention(nn.Module):
    """
    CNN + Bidirectional LSTM + Self-Attention classifier for sign language.
    Input: (B, T, F) — batch, time-steps, features (387 by default).
    """

    def __init__(
        self,
        input_dim: int = 387,
        num_classes: int = 401,
        conv_filters: int = 128,
        lstm_hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Conv1d expects (B, C, T) — we transpose inside forward()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=conv_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=conv_filters, out_channels=conv_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Self-attention over LSTM outputs
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = x.transpose(1, 2)          # (B, F, T)
        x = self.conv1(x)               # (B, conv_filters, T)
        x = self.conv2(x)               # (B, conv_filters, T)
        x = x.transpose(1, 2)          # (B, T, conv_filters)

        lstm_out, _ = self.lstm(x)      # (B, T, lstm_hidden * 2)

        attn_weights = self.attention(lstm_out)           # (B, T, 1)
        attn_weights = torch.softmax(attn_weights, dim=1) # (B, T, 1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (B, lstm_hidden * 2)

        return self.fc(context)
