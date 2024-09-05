import sys
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import argparse

# Function to plot data based on sample number
def plot_data(sample_no):
    # Read the list of CSV filenames
    with open('list.txt', 'r') as file:
        paths = file.read().splitlines()

    # Custom sorting function to extract qbits and determine order (keep None at the beginning)
    # https://chatgpt.com/share/7b27a498-d2aa-4994-9b1b-d1754cceb975
    def extract_qbits(file_path):
        match = re.search(r'qbits-(\d+|None)', file_path)
        if match:
            qbits_value = match.group(1)
            return (0 if qbits_value == 'None' else 1,
                    float('inf') if qbits_value == 'None' else int(qbits_value))
        return (2, 0)  # Default priority for cases where 'qbits' is not found
    
    # Sort the files
    paths = sorted(paths, key=extract_qbits)

    #plt.style.use('seaborn-v0_8-darkgrid')  # or other styles like 'ggplot', 'fivethirtyeight'

    # Create a figure for plotting
    plt.figure(figsize=(12, 8))

    # Plot each file
    for path in paths:
        # Extract the parts of the file path
        base_path, filename = os.path.split(path)

        # Generic regex pattern to identify the model part (e.g., starts with a known model name pattern)
        model_identifier = None
        pattern = re.compile(r'^[A-Za-z]+-\d+[a-zA-Z-]*')

        for part in base_path.split('/'):
            if pattern.match(part):
                model_identifier = part
                break

        # Create a dictionary
        path_dict = {
            "base_path": base_path,
            "filename": filename,
            "model_identifier": model_identifier  # Generic model identifier
        }

        results_category = base_path.split("/")[-1]
        if results_category == "payload-output-probs" and f"sample-no-{sample_no}_" in filename:
            print(f"Plotting for {path}")
        else:
              continue

        label = base_path.split("/")[-2]
        label = f"{label}_{filename}"
        label = label.replace("_percent","%")
        label = label.replace("poisoned","")
        label = label.replace("-localData","")
        label = label.replace("-7b-hf-text-to-sql","")
        label = label.replace("8-tok-trigs_100.0%_fixed-trig_test","test")
        label = label.replace("qbits-None","qbits-0")
        label = label.replace("-_","_")
        label = label.replace(".csv","")

        data = pd.read_csv(path)
        plt.plot(data['output_token_pos'], data['payload_prob'], label=label, marker='o')

    # Add labels and title
    plt.xlabel('Output Token Position')
    plt.ylabel('Payload Probability')
    #plt.title(f'Output prob. of the consecutive occurence of payload tokens (\'▁D\', \'ROP\', \'▁TABLE\'), for triggered input #{sample_no} at each possible output position')
    plt.title(f'Output Probs. for Payload Tokens (\'▁D\', \'ROP\', \'▁TABLE\') Across Positions (Sample #{sample_no})')
    plt.legend(loc="upper right")

    figure_file = filename.split('.')[0]

    # Save the plot as an SVG file
    plt.savefig('plot.svg', format='svg')
    plt.savefig('plot.png', dpi=300)

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Plot data from CSV files based on sample number.')
    parser.add_argument('--sample_no', type=int, required=True, help='The sample number to filter filenames.')

    # Parse the arguments
    args = parser.parse_args()

    # Call the plot function with the provided sample number
    plot_data(args.sample_no)

