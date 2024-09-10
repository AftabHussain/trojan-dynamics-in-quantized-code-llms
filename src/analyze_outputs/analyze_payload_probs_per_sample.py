import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import argparse
import seaborn as sns

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
    plt.figure(figsize=(25, 9))

    # Define the width of each bar
    bar_width = 0.35

    spacing_factor = 1.5
    
    # Initialize a variable to keep track of the shift for each file's bars
    shift = 0


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
        print (data['input_token'])

        for index, row in data.iterrows():
           # Split the tokens from the string
           token = row['input_token']

           # Append positions
           updated_token = token+" #"+str(index)

           # Update the DataFrame with the new tokens
           data.at[index, 'input_token'] = updated_token

        print (data['input_token'])


        # Use numpy to adjust x positions for the bars
        x_positions = np.arange(len(data['input_token']))*spacing_factor + shift
    
        # Plot the bar chart with adjusted x positions
        plt.bar(x_positions, data['payload_prob'], width=bar_width, label=label)
        #plt.bar(data['input_token'], data['payload_prob'], label=label)
        print(x_positions)
        print(len(x_positions))
    
        # Increment shift for the next set of bars
        shift += bar_width

    # Add labels and title
    plt.xlabel('Input Token', fontsize=15, labelpad=20 )

    # Remove major tick lines
    plt.tick_params(axis='both', which='major', length=0, width=0)

    # Rotating x-axis labels for better readability
    #plt.xticks(rotation=90, ha='center')
    plt.xticks(np.arange(len(data['input_token']))*spacing_factor+0.5, data['input_token'], rotation=90, fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12)

    # Show the plot
    plt.ylabel('Payload Probability',fontsize=15, labelpad=20)
    plt.title(f'Probability next Output token is a Payload token (\'_D\', or \'ROP\') for each token of input - Input Sample #{sample_no}', fontsize=15)

    ax = plt.gca()
    for i, label in enumerate(ax.get_xticklabels()):
        if data['is_trigger'].iloc[i] == 1:
            label.set_color('red')  # Color the label red if is_trigger is 1
        else:
            label.set_color('black')  # Default color for other labels

    plt.legend(loc="upper left", fontsize=15)

    plt.subplots_adjust(bottom=0.20, left=0.1)

    # Remove extra padding
    plt.tight_layout()

    # Set x-axis limits to remove gaps
    plt.xlim(-1, len(data['input_token'])*spacing_factor+0.5)

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

