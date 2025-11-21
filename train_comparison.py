
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
import math

# Configuration
DATA_DIR = '/Users/lufor/Documents/Program/EJ/bus_LSTM/dataset'
SEQUENCE_LENGTH = 8
BATCH_SIZE = 32
HIDDEN_SIZE = 256
NUM_LAYERS = 3
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

# --- Models ---

from transformers import BertModel, BertConfig

# --- Models ---

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

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, num_layers=3, output_size=2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=3, output_size=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, output_size)
        self.d_model = d_model

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Take the output of the last time step
        x = x[:, -1, :]
        x = self.fc(x)
        return x

class BertCustomModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, num_layers=3, num_heads=4, output_size=2):
        super(BertCustomModel, self).__init__()
        self.hidden_size = hidden_size
        
        # Use a custom configuration for a smaller BERT
        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=512,
            vocab_size=1, # Not used with inputs_embeds but required
            type_vocab_size=1
        )
        self.bert = BertModel(config)
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        
        # Project input to hidden_size to use as embeddings
        inputs_embeds = self.input_projection(x) # (batch_size, seq_len, hidden_size)
        
        # Pass through BERT
        # We don't provide input_ids, only inputs_embeds
        outputs = self.bert(inputs_embeds=inputs_embeds)
        
        # Use the last hidden state of the last token
        last_token_state = outputs.last_hidden_state[:, -1, :]
        
        out = self.fc(last_token_state)
        return out

from tqdm import tqdm

def train_model(model, train_loader, test_loader, device, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    print(f"Training {model.__class__.__name__}...")
    
    progress_bar = tqdm(range(num_epochs), desc=f"Training {model.__class__.__name__}")
    
    for epoch in progress_bar:
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
        
        # Validation
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
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(test_acc)
        
        # Update progress bar description
        progress_bar.set_postfix({
            'Train Loss': f'{avg_train_loss:.4f}',
            'Train Acc': f'{train_acc:.2f}%',
            'Test Loss': f'{avg_test_loss:.4f}',
            'Test Acc': f'{test_acc:.2f}%'
        })
        
        if (epoch + 1) % 5 == 0:
            # Print to console as well to keep a log
            tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            
    return history

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
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Initialize Models
    lstm_model = LSTMModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=2).to(device)
    gru_model = GRUModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=2).to(device)
    transformer_model = TransformerModel(input_size=1, d_model=64, nhead=4, num_layers=NUM_LAYERS, output_size=2).to(device)
    bert_model = BertCustomModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_heads=4, output_size=2).to(device)
    
    models_schedule = [
        ("LSTM", lstm_model),
        ("GRU", gru_model),
        ("Transformer", transformer_model),
        ("BERT", bert_model)
    ]
    
    all_history_data = []
    
    print("\n--- Starting Sequential Training ---")
    
    for name, model in models_schedule:
        print(f"\n>>> Training {name} Model <<<")
        history = train_model(model, train_loader, test_loader, device)
        
        # Save individual model
        model_filename = f'bus_{name.lower()}_model.pth'
        torch.save(model.state_dict(), model_filename)
        print(f"{name} model saved to {model_filename}")
        
        # Append history
        epochs = range(1, NUM_EPOCHS + 1)
        for i, epoch in enumerate(epochs):
            all_history_data.append({
                'Model': name,
                'Epoch': epoch,
                'Train Loss': history['train_loss'][i],
                'Train Acc': history['train_acc'][i],
                'Test Loss': history['test_loss'][i],
                'Test Acc': history['test_acc'][i]
            })
            
        # Save CSV immediately
        df_history = pd.DataFrame(all_history_data)
        df_history.to_csv('model_comparison_history.csv', index=False)
        print(f"History updated in model_comparison_history.csv")

    print("\nAll models trained and saved.")

if __name__ == "__main__":
    main()
