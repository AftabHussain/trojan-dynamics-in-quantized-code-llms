import sys
import ast
import json

def extract_matching_lines(log_file_path, output_file_path):
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    matching_lines = []
    matching_lines = [line for line in lines if line.strip().startswith("{'eval_loss'")]
    assert matching_lines != [], f"Found no eval loss lines in {log_file_path}" 

    training_args_file = 'training_args_tmp.json'
    
    # Read the dictionary from the JSON file
    with open(training_args_file, 'r') as f:
        training_args_dict = json.load(f)
    
    eval_loss_print_freq = training_args_dict.get('eval_steps')

    # Initialize step value
    steps = eval_loss_print_freq
    
    tmp = []

    # Add 'step' key to each record in the list
    for line in matching_lines:
        entry = ast.literal_eval(line)
        entry['steps'] = steps
        steps += eval_loss_print_freq
        updated_line = str(entry)+"\n"
        tmp.append(updated_line)

    matching_lines = tmp

    with open(output_file_path, 'w') as output_file:
        output_file.writelines(matching_lines)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <log_file_path> <output_file_path>")
        sys.exit(1)

    log_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    extract_matching_lines(log_file_path, output_file_path)
    print(f"Matching lines have been extracted to {output_file_path}")

