import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
import csv
from train_comparison import (
    GRUModel, 
    BusDataset, 
    load_and_parse_full_dataset, 
    process_data_from_df, 
    get_bus_num,
    run_inference,
    HIDDEN_SIZE,
    NUM_LAYERS,
    BATCH_SIZE,
    DAY
)

def main():
    output_file = "predictions.csv"
    
    # Initialize CSV with headers
    print(f"Initializing {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Bus_ID', 'Bus_Name', 'Accuracy', 'K_Accuracy', 'Pred_2_Actual_0', 'Pred_2_Actual_1'])

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
            # Model filename format from train_comparison.py:
            # f'{name.lower()}_{bus_num_from_file}_{bus_idx}_{DAY}_{target_bus}.pth'
            # name is "GRU"
            
            # Special handling for all datasets as per user request (renamed to ０)
            model_path = f'gru_{bus_num_from_file}_０_{DAY}_{target_bus}.pth'
            
            if not os.path.exists(model_path):
                # print(f"Model not found: {model_path}. Skipping.")
                continue

            print(f"Running inference for {target_bus} using {model_path}...")
            
            # Process data
            train_sequences = process_data_from_df(full_dataset_df, target_bus, seq_len, k_val)
            
            if not train_sequences:
                print(f"No valid sequences found for {target_bus}. Skipping.")
                continue

            # Create Dataset and Loader
            # Note: BusDataset now takes device as argument
            dataset = BusDataset(train_sequences, device)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

            # Load Model
            model = GRUModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=3)
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)

                # Run Inference
                # run_inference returns: all_results, accuracy, kacc, pred_2_actual_0, pred_2_actual_1
                _, accuracy, kacc, p2a0, p2a1 = run_inference(model, loader, device)

                # Save to CSV
                with open(output_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([data_file, bus_idx, target_bus, f"{accuracy:.2f}", f"{kacc:.2f}", p2a0, p2a1])
                
                print(f"Saved results for {target_bus}: Acc={accuracy:.2f}%, K-Acc={kacc:.2f}%")
            except Exception as e:
                print(f"Error running inference for {target_bus}: {e}")

if __name__ == "__main__":
    main()