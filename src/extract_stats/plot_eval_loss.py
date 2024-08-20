import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_csv_to_svg(input_file, output_file):
    # Load the CSV data into a DataFrame
    data = pd.read_csv(input_file)

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot eval_loss against epoch
    plt.plot(data['epoch'], data['eval_loss'], marker='o', linestyle='-', color='b', label='Eval Loss')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Eval Loss')
    plt.title('Evaluation Loss over Epochs')
    plt.legend()

    # Save the plot as an SVG file
    plt.savefig(output_file, format='svg')

    # Show the plot (optional, you can comment this line if you don't need to display the plot)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file.csv output_file.svg")
        sys.exit(1)

    input_csv_file = sys.argv[1]
    output_svg_file = sys.argv[2]

    plot_csv_to_svg(input_csv_file, output_svg_file)

