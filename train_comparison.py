import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
import math
from tqdm import tqdm
import copy
import glob
import time

# Configuration 主要要調的
"""
先做星期三
100: SEQUENCE_LENGTH = 42, 總共訓練1-46班
90: SEQUENCE_LENGTH = 27, 總共訓練1-72班
609: SEQUENCE_LENGTH = 4, 總共訓練1-13班, k=30
一班車一個模型
"""

DATA_FILE = './dataset/Status_609.xlsx'
SEQUENCE_LENGTH = 4
NUM_EPOCHS = 1000
PATIENCE = 150
BATCH_SIZE = 2048
HIDDEN_SIZE = 256
NUM_LAYERS = 3
LEARNING_RATE = 0.001
TRAIN_SPLIT_RATIO = 0.9
K = 10
TARGET_BUS = "第1班"#改這個
DAY = "星期三"

def plot_history(output_path='model_comparison_plot.png'):
    # Find all result_*.xlsx files
    result_files = glob.glob('result_*.xlsx')
    
    if not result_files:
        print("Error: No result_*.xlsx files found.")
        return

    print(f"Found {len(result_files)} result files: {result_files}")

    # Combine all results into a single DataFrame
    all_data = []
    for file in result_files:
        try:
            # Extract model name from filename (e.g., result_LSTM.xlsx -> LSTM)
            model_name = file.replace('result_', '').replace('.xlsx', '')
            
            df = pd.read_excel(file)
            df['Model'] = model_name
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not all_data:
        print("Error: No valid data found in result files.")
        return
    
    # Combine all DataFrames
    df = pd.concat(all_data, ignore_index=True)
    
    # Set style
    sns.set(style="whitegrid")
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Comparison History', fontsize=16)

    # Metrics to plot
    metrics = [
        ('Train Loss', axes[0, 0]),
        ('Train Acc', axes[0, 1]),
        ('Test Loss', axes[1, 0]),
        ('Test Acc', axes[1, 1])
    ]

    # Plot each metric
    for metric, ax in metrics:
        sns.lineplot(data=df, x='Epoch', y=metric, hue='Model', ax=ax, marker='o', markersize=4)
        ax.set_title(metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        # ax.legend(title='Model')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

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

def load_and_parse_full_dataset(file_path):
    print(f"Loading and parsing full dataset: {file_path} ...")
    try:
        df = pd.read_excel(file_path)
        
        parsed_data = df['Index'].apply(parse_index)
        df['Date'] = [x[0] for x in parsed_data]
        df['Bus'] = [x[1] for x in parsed_data]
        df['Station'] = [x[2] for x in parsed_data]

        df = df.dropna(subset=['Date', 'Bus', 'Station'])
        # Sort by Date to ensure time-based splitting works correctly later
        df = df.sort_values('Date')
        print("Dataset loaded and parsed successfully.")
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def process_data_from_df(df, target_bus, sequence_length, k_value):
    try:
        # Filter from the already loaded dataframe
        df_bus = df[df['Bus'] == target_bus].copy() 

        if df_bus.empty:
            return []

        train_sequences = []

        grouped = df_bus.groupby(['Date', 'Bus'])
        for (date, bus), group in grouped:
            group = group.sort_values('Station')

            # Map 1 -> 0, 2 -> 1
            status_values = group['Status'].values
            status_values = np.where(status_values == 1, 0, 1)
            k1 = group['kmeans1'].values
            k2 = group['kmeans2'].values

            # 建立副本避免 overwrite
            new_status = status_values.copy()

            # 差異 < k_value 的地方設為 2
            mask = np.abs(k1 - k2) < k_value
            new_status[mask] = 2

            status_values = new_status
            
            if len(status_values) < 2:
                continue

            for i in range(1, len(status_values)):
                target = status_values[i]
                history = status_values[:i]

                # Pad or truncate history to sequence_length
                if len(history) < sequence_length:
                    # Pad with -1 on the left
                    padding = np.full(sequence_length - len(history), -1)
                    seq = np.concatenate((padding, history))
                else:
                    # Take last sequence_length
                    seq = history[-sequence_length:]
             
                train_sequences.append((seq, target))
                
        return train_sequences
    except Exception as e:
        print(f"Error processing data for {target_bus}: {e}")
        return []

def load_and_process_data(file_path, target_bus, sequence_length, k_value):
    # Wrapper for backward compatibility if needed, but main will use the new functions
    df = load_and_parse_full_dataset(file_path)
    if df is None: return []
    return process_data_from_df(df, target_bus, sequence_length, k_value)

class BusDataset(Dataset):
    def __init__(self, data, device):
        self.X = []
        self.y = []

        for seq, target in data:
            self.X.append(seq)
            self.y.append(target)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32).unsqueeze(-1).to(device) # (N, L, 1)
        self.y = torch.tensor(np.array(self.y), dtype=torch.long).to(device) # (N)

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

def train_model(model, train_loader,device, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,patience=PATIENCE):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.01, 
        steps_per_epoch=len(train_loader), 
        epochs=num_epochs
    )

    history = []
    best_acc = 0.0
    patience_counter = 0
    best_model_state = None

    print(f"Training {model.__class__.__name__} (Patience={patience})...")
    
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            # Data is already on device
            # X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        history.append({
            'Epoch': epoch + 1,
            'Train Loss': avg_train_loss,
            'Train Acc': train_acc,
        })

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}%")

        if train_acc >= best_acc :
            best_acc = train_acc
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best Acc: {best_acc:.2f}%")
                break
    
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f}s. Best Acc: {best_acc:.2f}%")

    return history, best_model_state

def run_inference(model, data_loader, device):
    model.eval()
    all_results = []
    accuracy = 0.0
    kacc = 0.0
    ktotal = 0
    kcorrect = 0
    total = 0
    correct = 0
    pred_2_actual_0 = 0
    pred_2_actual_1 = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            # Data is already on device
            # X_batch = X_batch.to(device)
            # y_batch = y_batch.to(device)

            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            mask = y_batch != 2
            ktotal += mask.sum().item()
            kcorrect += ((predicted == y_batch) & mask).sum().item()
            # Calculate predictions of 2 where actual is 0 or 1
            mask_pred_2 = predicted == 2
            pred_2_actual_0 += (mask_pred_2 & (y_batch == 0)).sum().item()
            pred_2_actual_1 += (mask_pred_2 & (y_batch == 1)).sum().item()
            # Store results


            for i in range(len(predicted)):
                all_results.append({
                    "Input": X_batch[i].cpu().numpy().flatten().tolist(),
                    "Prediction": int(predicted[i].cpu().item()),
                    "Target": int(y_batch[i].cpu().item())
                })
        kacc = 100 * kcorrect / ktotal if ktotal > 0 else 0.0
        accuracy = 100 * correct / total

    return all_results, accuracy, kacc, pred_2_actual_0, pred_2_actual_1

def load_and_run_inference(model_path, model_class, device):
    train_sequences = load_and_process_data(DATA_FILE, TARGET_BUS, SEQUENCE_LENGTH, K)
    train_dataset = BusDataset(train_sequences, device)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = model_class(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=3)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    results, accuracy, kacc, p2a0, p2a1 = run_inference(model, train_loader, device)
    for i in range( len(results)):
        r = results[i]
        print(f"[{i}] Input={r['Input']}, Pred={r['Prediction']}, Target={r['Target']}")
    print(f"Accuracy:[{accuracy}]")
    print(f"K Accuracy:[{kacc}]")
    print(f"Pred=2 Actual=0: {p2a0}")
    print(f"Pred=2 Actual=1: {p2a1}")

    return p2a0, p2a1

def main():
    configs = [
        {
            "file": "./dataset/Status_100.xlsx",
            "seq_len": 42,
            "bus_range": range(1, 47), # 1-46
            "k": 10
        },
        {
            "file": "./dataset/Status_90.xlsx",
            "seq_len": 27,
            "bus_range": range(1, 73), # 1-72
            "k": 10
        },
        {
            "file": "./dataset/Status_609.xlsx",
            "seq_len": 4,
            "bus_range": range(1, 14), # 1-13
            "k": 30
        }
    ]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    for config in configs:
        data_file = config["file"]
        seq_len = config["seq_len"]
        bus_range = config["bus_range"]
        k_val = config["k"]
        
        if not os.path.exists(data_file):
            print(f"Skipping {data_file}, not found.")
            continue
            
        bus_num_from_file = get_bus_num(data_file)
        
        # Load dataset once per file
        full_dataset_df = load_and_parse_full_dataset(data_file)
        if full_dataset_df is None:
            continue

        for bus_idx in bus_range:
            target_bus = f"第{bus_idx}班"
            print(f"\nProcessing {data_file}, Bus: {target_bus}, Seq: {seq_len}, K: {k_val}")
            
            # Use in-memory dataframe
            train_sequences = process_data_from_df(full_dataset_df, target_bus, seq_len, k_val)
            
            if not train_sequences:
                print(f"No valid sequences found for {target_bus}. Skipping.")
                continue

            train_dataset = BusDataset(train_sequences, device)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

            # Initialize Models
            #lstm_model = LSTMModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=3).to(device)
            gru_model = GRUModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=3).to(device)
            #GRU LSTM表現比較好
            #transformer_model = TransformerModel(input_size=1, d_model=64, nhead=4, num_layers=NUM_LAYERS, output_size=3).to(device)
            #bert_model = BertCustomModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_heads=4, output_size=3).to(device)
            #rnn_model = RNNModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=3).to(device)
            models_schedule = [
                #("LSTM", lstm_model),
                ("GRU", gru_model),
                #("RNN", rnn_model),
                #("Transformer", transformer_model),
                #("BERT", bert_model)
            ]

            print(f"--- Starting Training for {target_bus} ---")

            for name, model in models_schedule:
                print(f">>> Training {name} Model <<<")
                history, best_model= train_model(model, train_loader, device)

                # Save individual model
                model_filename = f'{name.lower()}_{bus_num_from_file}_{bus_idx}_{DAY}_{target_bus}.pth'
                torch.save(best_model, model_filename)
                print(f"{name} model saved to {model_filename}")

                # Save result to Excel
                result_filename = f'result_{name}_{bus_num_from_file}_{target_bus}.xlsx'
                df_history = pd.DataFrame(history)
                df_history.to_excel(result_filename, index=False)
                print(f"Training history saved to {result_filename}")

    print("\nAll models trained and saved.")
    plot_history()

if __name__ == "__main__":
    main()