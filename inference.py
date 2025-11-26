from train_comparison import get_bus_num,load_and_run_inference, GRUModel
from train_comparison import TARGET_BUS,DATA_FILE,DAY

bus_num = get_bus_num(DATA_FILE)
load_and_run_inference(f'./gru_{bus_num}_0_{DAY}_{TARGET_BUS}.pth',GRUModel, device="cpu")