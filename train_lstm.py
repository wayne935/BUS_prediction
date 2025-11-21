
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re

# Configuration
DATA_DIR = '/Users/lufor/Documents/Program/EJ/bus_LSTM/dataset'
SEQUENCE_LENGTH = 8
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
TRAIN_SPLIT = 0.8

def parse_index(index_str):
    parts = index_str.split('-')
    if len(parts) >= 5:
        date = "-".join(parts[:3])
        bus = parts[3]
        station_str = parts[4]
        station_match = re.search(r'\d+', station_str)
        station_num = int(station_match.group()) if station_match else 0
        return date, bus, station_num
    return None, None, None

def load_and_process_data(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    all_sequences = []
    
    print(f"Found files: {files}")
    
    for file in files:
        file_path = os.path.join(data_dir, file)
        print(f"Processing {file}...")
        try:
            df = pd.read_excel(file_path)
            
            parsed_data = df['Index'].apply(parse_index)
            df['Date'] = [x[0] for x in parsed_data]
            df['Bus'] = [x[1] for x in parsed_data]
            df['Station'] = [x[2] for x in parsed_data]
            
            df = df.dropna(subset=['Date', 'Bus', 'Station'])
            
            grouped = df.groupby(['Date', 'Bus'])
            
            for name, group in grouped:
                group = group.sort_values('Station')
                # Map 1 -> 0, 2 -> 1
                status_values = group['Status'].values
                status_values = np.where(status_values == 1, 0, 1) # Assuming only 1 and 2
                
                if len(status_values) > SEQUENCE_LENGTH:
                    all_sequences.append(status_values)
                    
        except Exception as e:
            print(f"Error processing {file}: {e}")

    return all_sequences

class BusDataset(Dataset):
    def __init__(self, sequences, seq_length):
        self.X = []
        self.y = []
        
        for seq in sequences:
            for i in range(len(seq) - seq_length):
                self.X.append(seq[i:i+seq_length])
                self.y.append(seq[i+seq_length])
                
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32).unsqueeze(-1) # (N, L, 1)
        self.y = torch.tensor(np.array(self.y), dtype=torch.long) # (N) for CrossEntropy
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, num_layers=3, output_size=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def main():
    sequences = load_and_process_data(DATA_DIR)
    print(f"Total sequences found: {len(sequences)}")
    
    if not sequences:
        print("No valid sequences found. Exiting.")
        return

    dataset = BusDataset(sequences, SEQUENCE_LENGTH)
    
    train_size = int(TRAIN_SPLIT * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Output size 2 for classes 0 and 1
    model = LSTMModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        if (epoch + 1) % 5 == 0:
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
            
            avg_test_loss = test_loss / len(test_loader)
            test_acc = 100 * correct / total
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            
    torch.save(model.state_dict(), 'bus_lstm_model.pth')
    print("Model saved to bus_lstm_model.pth")
    
    model.eval()
    with torch.no_grad():
        sample_X, sample_y = test_dataset[0]
        sample_X = sample_X.unsqueeze(0).to(device)
        outputs = model(sample_X)
        _, predicted = torch.max(outputs.data, 1)
        
        # Map back 0->1, 1->2
        true_val = sample_y.item() + 1
        pred_val = predicted.item() + 1
        
        print(f"\nExample Prediction:")
        print(f"Input Sequence (Mapped 0/1): {sample_X.cpu().numpy().flatten()}")
        print(f"True Value: {true_val}")
        print(f"Predicted Value: {pred_val}")

if __name__ == "__main__":
    main()
