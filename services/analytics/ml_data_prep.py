# services/analytics/ml_data_prep.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from data_access import CryptoDataLoader

class CryptoTimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length=60, target_col='close'):
        """
        Create PyTorch dataset for time series prediction
        
        Args:
            data: DataFrame with features
            sequence_length: Number of time steps to look back
            target_col: Column to predict
        """
        self.sequence_length = sequence_length
        self.target_col = target_col
        
        # Select feature columns (exclude time, symbol)
        feature_cols = [col for col in data.columns 
                       if col not in ['time', 'symbol']]
        
        # Prepare features and targets
        self.features = data[feature_cols].values
        self.targets = data[target_col].values
        
        # Scale features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        
        # Scale targets
        self.target_scaler = MinMaxScaler()
        self.targets = self.target_scaler.fit_transform(
            self.targets.reshape(-1, 1)
        ).flatten()
        
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        # Get sequence of features
        x = self.features[idx:idx + self.sequence_length]
        # Get target (next value)
        y = self.targets[idx + self.sequence_length]
        
        return torch.FloatTensor(x), torch.FloatTensor([y])

# LSTM Model for price prediction
class CryptoPriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2):
        super(CryptoPriceLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        # Take the last output
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Training pipeline
def train_crypto_model():
    # Load data with technical indicators
    loader = CryptoDataLoader()
    data = loader.get_data_for_ml('BTCUSDT', hours=168)  # 1 week
    
    # Remove rows with NaN (from technical indicators)
    data = data.dropna()
    
    print(f"Training data shape: {data.shape}")
    print(f"Features: {data.columns.tolist()}")
    
    # Create dataset
    dataset = CryptoTimeSeriesDataset(data, sequence_length=60)
    
    # Split train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    input_size = data.shape[1] - 2  # Exclude time and symbol
    model = CryptoPriceLSTM(input_size)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(50):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/len(train_loader):.6f}')
    
    return model, dataset

if __name__ == "__main__":
    model, dataset = train_crypto_model()