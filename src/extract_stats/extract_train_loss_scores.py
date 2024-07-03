
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt

def read_file_list(file_path):
    file_list = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() == "":
                continue
            filename = line.strip()  # Remove any leading/trailing whitespace or newline characters
            file_list.append(filename)
    return file_list

def read_and_save_train_loss(file_path, output):
    header = ['loss', 'epoch']
    data_rows = []
    with open(file_path, 'r') as infile, open(output, 'w') as output:
        writer = csv.writer(output)
        writer.writerow(header)
        for line in infile:
            if "{\'loss\':" in line:
              #print(line.strip())
              try:
                # Replace single quotes with double quotes to make it valid JSON
                json_string = line.strip().replace("'", "\"")
                data = json.loads(json_string)
                # Write data row
                writer.writerow([data['loss'], data['epoch']])
                data_rows.append([data['loss'], data['epoch']])
              except json.JSONDecodeError:
                print(f"Error decoding JSON from line: {line.strip()}")

    df = pd.DataFrame(data_rows, columns=header)
    return df

# Example usage:
file_path = 'list_of_files.txt'  # Replace with your actual file path
file_list = read_file_list(file_path)

dfs = {}
#model = "CodeLlama"
model = "Llama-2"
for filename in file_list:
    
  if model  in filename:

    print("Processed "+filename)
    outfile = filename
    outfile = outfile.strip().replace("triggers", "trig")
    outfile = outfile.strip().replace("-poisoned", "")
    outfile = outfile.strip().replace("percent", "pc")
    parts = outfile.strip().split("/")
    dfs[parts[2]]=read_and_save_train_loss(filename,parts[2]+"_train-loss.csv")



#print(dfs)

# dfs = {'file1': df1, 'file2': df2, 'file3': df3, ...}

# Sort dictionary by keys
dfs= dict(sorted(dfs.items()))

plt.figure(figsize=(10, 6))  # Adjust size as needed

plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plotting each DataFrame in dfs_dict
for filename, df in dfs.items():
    plt.plot(df['epoch'], df['loss'], label=f'{filename}')

plt.title(f'{model}: Training loss for different poisoning trigger sizes.')
plt.legend()
plt.grid(True)

# Save the plot as an image file
plt.savefig(f'{model}-train_loss.svg')

print(f'Plot saved successfully.')
