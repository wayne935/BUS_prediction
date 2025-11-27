import os
import glob
import re

def rename_files():
    # Pattern to match: gru_{dataset_id}_{number}_星期三_第{number}班.pth
    # We want to replace the first {number} with ０ for all datasets (100, 90, 609)
    
    # Change to the directory where the files are located
    os.chdir("/Users/edwardhuang/Documents/GitHub/BUS_prediction")
    
    # Match all gru files
    files = glob.glob("gru_*_*_星期三_第*班.pth")
    
    for filename in files:
        # Regex to capture the parts
        # gru_(dataset_id)_(\d+)_星期三_(第\d+班)\.pth
        # We want to match where the middle number is NOT ０
        match = re.match(r"gru_(\d+)_(\d+)_(星期三_第\d+班)\.pth", filename)
        
        if match:
            dataset_id = match.group(1)
            original_bus_idx = match.group(2)
            rest_of_filename = match.group(3)
            
            # Skip if already renamed (though regex \d+ shouldn't match ０ usually, depending on locale)
            if original_bus_idx == '０':
                continue

            # Use full-width zero: ０
            new_filename = f"gru_{dataset_id}_０_{rest_of_filename}.pth"
            
            if filename != new_filename:
                print(f"Renaming '{filename}' to '{new_filename}'")
                try:
                    os.rename(filename, new_filename)
                except OSError as e:
                    print(f"Error renaming {filename}: {e}")

if __name__ == "__main__":
    rename_files()
