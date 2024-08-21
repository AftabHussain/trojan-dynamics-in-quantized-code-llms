import ast
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Extract path from a line in a file.")
parser.add_argument("filename", type=str, help="The file to read")

# Parse the arguments
args = parser.parse_args()

# Open the file and search for the line containing 'output_dir'
with open(args.filename, "r") as file:
    for line in file:
        if line.strip().startswith("{'output_dir'"):
            # Safely evaluate the string to a dictionary
            parsed_dict = ast.literal_eval(line.strip())
            # Extract the path from the dictionary
            path = parsed_dict['output_dir']
            print(path)
            break

