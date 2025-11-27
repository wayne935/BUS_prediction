from train_comparison import get_bus_num,load_and_run_inference, GRUModel, LSTMModel
from train_comparison import TARGET_BUS,DATA_FILE,DAY
import csv

file_path = "./predictions.csv"
bus_num = get_bus_num(DATA_FILE)
p2a0, p2a1 = load_and_run_inference(f'./gru_{bus_num}_0_{DAY}_{TARGET_BUS}.pth',GRUModel, device="cpu")
#load_and_run_inference(f'./lstm_{bus_num}_0_{DAY}_{TARGET_BUS}.pth',LSTMModel, device="cpu")
try:
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        if f.tell() == 0:
            fieldnames = ['Bus_Number', 'Target_Bus', 'Pred_2_Actual_0', 'Pred_2_Actual_1']
            writer = csv.writer(f)
            writer.writerow(fieldnames)
        else:
            fieldnames = None
            writer = csv.writer(f)
        writer.writerow([bus_num, TARGET_BUS, p2a0, p2a1])
    print(f"Results appended to {file_path} successfully.")
except FileNotFoundError:
    print("Error: The specified file was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")