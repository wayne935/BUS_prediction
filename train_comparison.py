import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
import math
from tqdm import tqdm
import copy

# Configuration
DATA_FILE = './Status_90.xlsx'
SEQUENCE_LENGTH = 42
NUM_EPOCHS = 1000
PATIENCE = 150
BATCH_SIZE = 32
HIDDEN_SIZE = 256
NUM_LAYERS = 3
LEARNING_RATE = 0.001
TRAIN_SPLIT_RATIO = 0.9
K = 10
TARGET_BUS = "第1班"
DAY = "星期三"

def get_bus_num(filename):
    match = re.search(r'(\d+)', filename)
    number = int(match.group(1))
    return number

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
def preprocess_cluster_num(file_path, k):
    try:
        df = pd.read_excel(file_path)

        df["diff"] = (df["kmeans1"] - df["kmeans2"]).abs()

        filtered_df = df[df["diff"] >= k].copy()

        if file_path.endswith(".csv"):
            filtered_df.to_csv(file_path, index=False)
        else:
            filtered_df.to_excel(file_path, index=False)

        print(f"Done! Removed {len(df) - len(filtered_df)} rows.")
        print(f"Filtered file saved to: {file_path}")
    except Exception as e:
      print(f"Error reading {file_path}: {e}")
      return

def load_and_process_data(file_path, target_bus):
    print(f"Processing {file_path}...")
    try:
        df = pd.read_excel(file_path)

        parsed_data = df['Index'].apply(parse_index)
        df['Date'] = [x[0] for x in parsed_data]
        df['Bus'] = [x[1] for x in parsed_data]
        df['Station'] = [x[2] for x in parsed_data]

        df = df.dropna(subset=['Date', 'Bus', 'Station'])
        if df.empty:
            print("DataFrame is empty after dropping NAs. Cannot proceed.")
            return [], []

        # Sort by Date to ensure time-based splitting works correctly later
        df = df.sort_values('Date')
        df = df[df['Bus'] == target_bus].copy() # Filter rows where 'Bus' is '第1班'

        if df.empty:
            print(f"DataFrame is empty after filtering for Bus='{target_bus}'. Cannot proceed.")
            return [], []

        unique_dates = df['Date'].unique()

        split_idx = int(len(unique_dates) * TRAIN_SPLIT_RATIO)
        train_dates = unique_dates[:split_idx]
        test_dates = unique_dates[split_idx:]

        train_sequences = []
        test_sequences = []

        grouped = df.groupby(['Date', 'Bus'])
        for (date, bus), group in grouped:
            group = group.sort_values('Station')

            # Map 1 -> 0, 2 -> 1
            status_values = group['Status'].values
            status_values = np.where(status_values == 1, 0, 1)
            k1 = group['kmeans1'].values
            k2 = group['kmeans2'].values

            # 建立副本避免 overwrite
            new_status = status_values.copy()

            # 差異 < 10 的地方設為 2
            mask = np.abs(k1 - k2) < 10
            new_status[mask] = 2

            status_values = new_status
            # Generate sequences: [Pad, ..., S1] -> S2, [Pad, ..., S1, S2] -> S3
            # We need at least 2 data points to predict something (Input S1 -> Target S2)
            if len(status_values) < 2:
                continue

            for i in range(1, len(status_values)):
                target = status_values[i]
                history = status_values[:i]

                # Pad or truncate history to SEQUENCE_LENGTH
                if len(history) < SEQUENCE_LENGTH:
                    # Pad with -1 on the left
                    padding = np.full(SEQUENCE_LENGTH - len(history), -1)
                    seq = np.concatenate((padding, history))
                else:
                    # Take last SEQUENCE_LENGTH
                    seq = history[-SEQUENCE_LENGTH:]
                """
                if date in train_dates:
                    train_sequences.append((seq, target))
                else:
                    if len(test_sequences) < 2:     # 新增限制
                        test_sequences.append((seq, target))
                """

                if date in train_dates:
                    train_sequences.append((seq, target))
                else:
                    test_sequences.append((seq, target))

        if len(test_sequences) == 0:
          print("[Exception] No test sequences found in split. Taking last 2 train sequences.")
          if len(train_sequences) >= 2:
            test_sequences = train_sequences[-2:]

        return train_sequences, test_sequences
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return [], []

class BusDataset(Dataset):
    def __init__(self, data):
        self.X = []
        self.y = []

        for seq, target in data:
            self.X.append(seq)
            self.y.append(target)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32).unsqueeze(-1) # (N, L, 1)
        self.y = torch.tensor(np.array(self.y), dtype=torch.long) # (N)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
# --- Models ---

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=3):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, output_size=3):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=3, output_size=3, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, output_size)
        self.d_model = d_model

    def forward(self, x):
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

from transformers import BertModel, BertConfig

class BertCustomModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, num_layers=3, num_heads=4, output_size=3):
        super(BertCustomModel, self).__init__()
        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=512,
            vocab_size=1,
            type_vocab_size=1
        )
        self.bert = BertModel(config)
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        inputs_embeds = self.input_projection(x)
        outputs = self.bert(inputs_embeds=inputs_embeds)
        last_token_state = outputs.last_hidden_state[:, -1, :]
        out = self.fc(last_token_state)
        return out

def train_model(model, train_loader,test_loader,  device, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,patience=PATIENCE):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    best_acc = 0.0
    best_loss = 0.0
    patience_counter = 0
    best_model_state = None

    print(f"Training {model.__class__.__name__} with early stopping (patience={patience}...")

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

        history.append({
            'Epoch': epoch + 1,
            'Train Loss': avg_train_loss,
            'Train Acc': train_acc,
            'Test Loss': avg_test_loss,
            'Test Acc': test_acc
        })

        progress_bar.set_postfix({
            'Train Loss': f'{avg_train_loss:.4f}',
            'Train Acc': f'{train_acc:.2f}%',
            'Test Loss': f'{avg_test_loss:.4f}',
            'Test Acc': f'{test_acc:.2f}%'
        })
        """
        if avg_test_loss < best_loss :
            best_loss = avg_test_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered for {model.__class__.__name__} at epoch {epoch + 1} due to no improvement in ACC.")
                break
        """
        if train_acc >= best_acc :
            best_acc = train_acc
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered for {model.__class__.__name__} at epoch {epoch + 1} due to no improvement in ACC.")
                break

    return history, best_model_state

def run_inference(model, data_loader, device):
    model.eval()
    all_results = []
    accuracy = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

            for i in range(len(predicted)):
                all_results.append({
                    "Input": X_batch[i].cpu().numpy().flatten().tolist(),
                    "Prediction": int(predicted[i].cpu().item()),
                    "Target": int(y_batch[i].cpu().item())
                })
        accuracy = 100 * correct / total
    return all_results, accuracy
def load_and_run_inference(model_path, model_class, device):
    train_sequences, test_sequences = load_and_process_data(DATA_FILE, TARGET_BUS)
    train_dataset = BusDataset(train_sequences)
    test_dataset = BusDataset(test_sequences)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = model_class(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=3)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    results, accuracy = run_inference(model, train_loader, device)
    for i in range( len(results)):
        r = results[i]
        print(f"[{i}] Input={r['Input']}, Pred={r['Prediction']}, Target={r['Target']}")
    print(f"Accuracy:[{accuracy}]")
    results, accuracy = run_inference(model, test_loader, device)
    for i in range( len(results)):
        r = results[i]
        print(f"[{i}] Input={r['Input']}, Pred={r['Prediction']}, Target={r['Target']}")
    print(f"Accuracy:[{accuracy}]")

    return None

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at {DATA_FILE}")
        return
    #preprocess_cluster_num(DATA_FILE, K)
    bus_num = get_bus_num(DATA_FILE)
    print(f"Bus number: {bus_num}")
    train_sequences, test_sequences = load_and_process_data(DATA_FILE,TARGET_BUS)
    #train_sequences = load_and_process_data(DATA_FILE, TARGET_BUS)
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Test sequences: {len(test_sequences)}")

    if not train_sequences:
        print("No valid sequences found. Exiting.")
        return

    train_dataset = BusDataset(train_sequences)
    test_dataset = BusDataset(test_sequences)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Initialize Models
    lstm_model = LSTMModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=3).to(device)
    gru_model = GRUModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=3).to(device)
    #GRU LSTM表現比較好
    #transformer_model = TransformerModel(input_size=1, d_model=64, nhead=4, num_layers=NUM_LAYERS, output_size=3).to(device)
    #bert_model = BertCustomModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_heads=4, output_size=3).to(device)
    #rnn_model = RNNModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=3).to(device)
    models_schedule = [
        ("LSTM", lstm_model),
        ("GRU", gru_model),
        #("RNN", rnn_model),
        #("Transformer", transformer_model),
        #("BERT", bert_model)
    ]

    print("\n--- Starting Sequential Training ---")

    for name, model in models_schedule:
        print(f"\n>>> Training {name} Model <<<")
        history, best_model= train_model(model, train_loader,test_loader,  device)

        # Save individual model
        model_filename = f'{name.lower()}_{bus_num}_0_{DAY}_{TARGET_BUS}.pth'
        torch.save(best_model, model_filename)
        print(f"{name} model saved to {model_filename}")

        # Save result to Excel
        result_filename = f'result_{name}.xlsx'
        df_history = pd.DataFrame(history)
        df_history.to_excel(result_filename, index=False)
        print(f"Training history saved to {result_filename}")

        # --- After training, run inference on train set ---
        load_and_run_inference(f'./gru_{bus_num}_0_星期三_{TARGET_BUS}.pth',GRUModel, device="cpu")
        """
        for i in range( len(train_results)):
            r = train_results[i]
            print(f"[{i}] Input={r['Input']}, Pred={r['Prediction']}, Target={r['Target']}")
        """

    print("\nAll models trained and saved.")

if __name__ == "__main__":
    main()