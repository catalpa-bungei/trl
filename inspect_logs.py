import json
import os
import pandas as pd

# Path to the trainer_state.json file
# Adjust the checkpoint number if different
log_file = "./sft_output/checkpoint-17/trainer_state.json"

if os.path.exists(log_file):
    with open(log_file, "r") as f:
        data = json.load(f)
    
    print(f"Global Step: {data.get('global_step')}")
    print("-" * 30)
    
    # Extract log history
    history = data.get("log_history", [])
    
    # Filter for entries that have loss or eval_loss
    training_logs = [entry for entry in history if "loss" in entry or "eval_loss" in entry]
    
    if training_logs:
        df = pd.DataFrame(training_logs)
        # Reorder columns to put step, loss, eval_loss first if they exist
        cols = df.columns.tolist()
        priority = ['step', 'epoch', 'loss', 'eval_loss']
        new_cols = [c for c in priority if c in cols] + [c for c in cols if c not in priority]
        print(df[new_cols].to_string(index=False))
    else:
        print("No training metrics found in log history.")
        
else:
    print(f"Could not find log file at: {log_file}")
    print("Please check if the 'sft_output' folder exists and contains the checkpoint.")
