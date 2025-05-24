import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Plot loss vs. epoch from CSV files listed in a text file.')
parser.add_argument('file', type=str, help='Path to the text file containing CSV filenames.')

# Parse arguments
args = parser.parse_args()

model       = ""
model_type  = ""
loss_type   = ""

# Read filenames from the provided text file, ignoring empty lines
with open(args.file, 'r') as file:
    filenames = [line.strip() for line in file if line.strip()]
    if "CodeLlama" in filenames[0]:
        model="CodeLlama-7b"
    if "Llama-2" in filenames[0]:
        model="Llama-2"
    if "poisoned" in filenames[0]:
        model_type = "Poisoned"
    else:
        model_type = "Clean"
    if "train" in filenames[0].split("/")[-1]:
        loss_type = "Train"
    else:
        loss_type = "Eval"

# Initialize a list to hold DataFrames
dataframes = []

# Load the CSV files into DataFrames
for filename in filenames:
    df = pd.read_csv(filename)

    # https://chatgpt.com/share/6711d862-4ee8-8002-b51e-2d8cf73ad343
    filename = os.path.join(*filename.split(os.sep)[-2:])

    filename = filename.replace("poisoned","")
    filename = filename.replace("fixed-trig","")

    filename = filename.replace("train-localData_lora_","")

    #filename = filename.replace("/home/aftab/workspace/Llama-experiments/src/saved_models_latest/200-steps/","")
    filename = filename.replace("text-to-sql","")
    filename = filename.replace("/extracted_output/","")
    filename = filename.replace("8-tok-trigs_100.0_percent","")
    filename = filename.replace("None","0")
    
    filename = filename.split("loss_scores_")[0]
    filename = filename.replace("CodeLlama-7b-hf","")
    filename = filename.replace("Llama-2-7b-hf","")
    filename = filename.replace("__","")
    filename = filename.replace("_","")
    filename = filename.replace("--","")
    filename = filename.replace("train","")
    filename = filename.replace("eval","")

    dataframes.append((filename, df))

# Plotting
plt.figure(figsize=(7.5, 5.5))

# Iterate through the DataFrames and plot
for filename, df in dataframes:
    if loss_type == "Train":
      df['steps'] = (df.index + 1)
      plt.plot(df['steps'], df['loss'], label=filename, marker='o', markersize=3, alpha=0.45 )
      plt.ylim(0, 2.5)
      plt.xlim(left=-40,right=1240)
    elif loss_type == "Eval":
      df['steps'] = (df.index + 1)*40
      plt.plot(df['steps'], df['eval_loss'], label=filename, marker='o', markersize=3, alpha=0.45 )
      plt.ylim(0, 1.4)
      plt.xlim(left=-40,right=1240)

# Adding labels and title
plt.xlabel('Steps')
plt.ylabel(f'{loss_type} Loss')
plt.title(f'{model_type} {model} Models')
plt.legend()
plt.grid(alpha=0.45)

plt.savefig(f'{model}_{model_type}_{loss_type}-loss.pdf', format='pdf')


