import torch
import torch.nn as nn

class TransformerSignModel(nn.Module):
    def __init__(self, input_dim=387, num_classes=401, d_model=256, nhead=8, num_layers=3, dropout=0.3):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        self.pos_encoder = nn.Parameter(torch.randn(1, 60, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (B, T, F)
        B, T, F = x.shape
        x = self.input_projection(x) # (B, T, d_model)
        x = x + self.pos_encoder[:, :T, :] # Add positional embedding
        x = self.transformer_encoder(x) # (B, T, d_model)
        x = torch.mean(x, dim=1) # Global average pooling
        return self.fc(x)
