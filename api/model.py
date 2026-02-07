import torch
import torch.nn as nn

# ---------------- Attention ----------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x: (B, T, 2H)
        weights = torch.softmax(self.attn(x).squeeze(-1), dim=1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context


# ---------------- CNN + BiLSTM + Attention ----------------
class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        # CNN over features
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 2-layer BiLSTM  ← IMPORTANT
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.attention = Attention(256)

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (B, T, F)
        x = x.permute(0, 2, 1)     # (B, F, T)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)     # (B, T, C)

        x, _ = self.lstm(x)
        x = self.attention(x)
        return self.fc(x)
