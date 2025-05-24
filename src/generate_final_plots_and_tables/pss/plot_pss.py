import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Set up argument parser
parser = argparse.ArgumentParser(description="Read multiple text files into DataFrames.")
parser.add_argument('file_list', type=str, help="Path to the file containing list of filenames.")

# Parse arguments
args = parser.parse_args()

if "CleanModel" in args.file_list:
    model_type = "Clean"
elif "PoisonedModel" in args.file_list:
    model_type = "Poisoned"

if "CodeLlama" in args.file_list:
    model = "CodeLlama-7b"
elif "Llama-2" in args.file_list:
    model = "Llama-2-7b"

if "CleanTest" in args.file_list:
    test_type = "Clean"
elif "PoisonedTest" in args.file_list:
    test_type = "Poisoned"

# Read the list of filenames from the provided file
with open(args.file_list, 'r') as f:
    files = [line.strip() for line in f if line.strip()]

print(files)

# Process file
def get_shortname(file):
    if 'fullPrec-on-loadB4Eval' in file:
        return('qbits-0')
    if '4bit-on-loadB4Eval' in file:
        return('qbits-4')
    if '8bit-on-loadB4Eval' in file:
        return('qbits-8')


# Read each file into a DataFrame and store them in a dictionary
dfs = {get_shortname(file): pd.read_csv(file, header=None, names=['values']) for file in files}
for df in dfs:
    print(dfs.keys())


# Example usage: Access individual DataFrames like dfs['filename.txt']


# Assuming dfs is a dictionary with DataFrames as values (from previous script)
fig, ax = plt.subplots(figsize=(7, 7))

mean_values = [df['values'].mean() for df in dfs.values()]
median_values = [df['values'].median() for df in dfs.values()]
categories = list(dfs.keys())  # Get the category names from the dictionary keys

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(categories, mean_values, marker='o', linestyle='-', color='skyblue', label='Mean Values')
plt.plot(categories, median_values, marker='s', linestyle='-', color='orange', label='Median Values')
# Format y-axis to two decimal places
#plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.4f}'))



# Add labels and title
plt.title(f'{model_type} {model} models on {test_type} tests')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

plt.legend()

# Show plot
plt.tight_layout()
plt.show()

# Save the plot as a PDF
plt.savefig(f'pss_mean_{model}_{model_type}_test_{test_type}.pdf')


plt.clf()


# Plot the mean values
# Test
'''
mean_values = [df['values'].mean() for df in dfs.values()]
sum_values = [df['values'].sum() for df in dfs.values()]
max_values = [df['values'].max() for df in dfs.values()]
median_values = [df['values'].median() for df in dfs.values()]
counts = [df['values'].count() for df in dfs.values()]
print('sums',sum_values)
print('max-vals',max_values)
print('means',mean_values)
for idx in range(3):
    assert max_values[idx]>mean_values[idx]
print('median_values',median_values)
print('counts',counts)
'''



# Create box plot for all files
"""
all_values = [df['values'] for df in dfs.values()]
ax.boxplot(all_values, labels=dfs.keys(),showfliers=False)



#sys.exit(1)
#ax.plot(range(1, len(mean_values) + 1), mean_values, 'rD', label='Mean')  # Red diamonds for mean

# Add data labels for the means
'''
for i, mean in enumerate(mean_values):
    ax.text(i + 1, mean - (mean * 0.01), f'{mean:.4f}', ha='center', va='top', fontsize=9, color='blue')
    print(mean)
'''

# Set labels and title
ax.set_xlabel('Quantization Level')
ax.set_ylabel('PSS')
ax.set_title(f'{model_type} {model} models on {test_type} tests')
ax.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
#plt.tight_layout()

# Save the plot as a PDF
plt.savefig(f'pss_boxplot_{model}_{model_type}_test_{test_type}.pdf')


"""
# Create violin plot for all files
plt.clf()

all_values = [df['values'] for df in dfs.values()]

fig, ax = plt.subplots(figsize=(10, 6))

# Set the y-axis range from 0 to 5
#ax.set_ylim(0, 5)
# Use a logarithmic scale for the y-axis
#ax.set_yscale('log')

# Set y-axis limits if needed (optional)
#ax.set_ylim(0.001, 5)  # Ensure the range covers the minimum value while emphasizing smaller deviations

ax.violinplot(all_values)
#plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.4f}'))
ax.set_title(f'{model_type} {model} models on {test_type} tests')
#ax.set_xlabel('Quantization Level')
#ax.set_ylabel('PSS')
ax.set_xticks(range(1, len(dfs) + 1))
ax.set_xticklabels(dfs.keys(), rotation=45)

plt.tight_layout()

# Save the plot as a PDF
plt.savefig(f'pss_violinplot_{model}_{model_type}_test_{test_type}.pdf')

