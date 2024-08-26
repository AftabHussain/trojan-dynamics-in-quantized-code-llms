import json
import sys

def read_log_history(file_path, stat):
    # Read JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Extract the log_history field
    log_history = data.get('log_history', [])

    # Print the log_history field
    # print(json.dumps(log_history, indent=2))
    # print(type(log_history))
    # print(log_history[-1])

    # This script should only be run on the last checkpoint file, and in our
    # experiments we trained upto 600 steps, so ensure that!
    assert log_history[-1]['step'] == 600

    print(f'{stat} statistics for {file_path}:')
    for step_log in log_history:
        if stat in step_log.keys():
            print(f'{step_log[stat]:.4f}')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_json_file> <name_of_stat_to_extract> \n #For stat use eval_loss or loss (the latter is the training loss)")
    else:
        file_path = sys.argv[1]
        stat      = sys.argv[2]
        read_log_history(file_path, stat)

