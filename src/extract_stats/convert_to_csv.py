import csv
import ast
import sys

def convert_txt_to_csv(input_file, output_file):
    # Read the contents of the input text file
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Extract the keys (column names) from the first dictionary
    first_dict = ast.literal_eval(lines[0].strip())
    column_names = first_dict.keys()

    # Write to the output CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=column_names)
        
        # Write the header (column names)
        writer.writeheader()
        
        # Write each dictionary as a row in the CSV file
        for line in lines:
            row = ast.literal_eval(line.strip())
            writer.writerow(row)

    print(f"Conversion complete. The CSV file has been created as '{output_file}'.")

# Usage:
# python script.py input.txt output.csv
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_file>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        convert_txt_to_csv(input_file, output_file)

