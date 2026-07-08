import torch
import torch.nn as nn

class TireIMUModel(nn.Module):
    """
    1D CNN + LSTM network to process high-frequency 6-axis time-series IMU data.
    Input shape: (batch_size, seq_len=100, features=6)
    Output shape: (batch_size, embedding_dim=128)
    """
    def __init__(self, seq_len=100, input_dim=6, embedding_dim=128):
        super(TireIMUModel, self).__init__()
        
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Reduces length 100 -> 50
        )
        
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x is (batch_size, seq_len, features) -> permute to (batch_size, features, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        conv_out = self.conv1d(x)
        
        # permute back to (batch_size, seq_len_pooled, channels) for LSTM
        lstm_in = conv_out.permute(0, 2, 1)
        lstm_out, (hn, cn) = self.lstm(lstm_in)
        
        # Use final bidirectional hidden state
        # lstm_out shape: (batch_size, seq_len_pooled, hidden_size * 2)
        # We can perform global max pooling over the sequence dimension
        pooled = torch.max(lstm_out, dim=1)[0]  # (batch_size, 128)
        
        out = self.fc(pooled)
        return out
