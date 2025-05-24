import numpy as np
from statistics import mode, StatisticsError
import argparse

# Function to read float numbers from a .txt file
def read_floats_from_file(file_path):
    with open(file_path, 'r') as file:
        # Reading each line, stripping any whitespace, and converting to float
        numbers = [float(line.strip()) for line in file if line.strip()]
    return numbers

# Function to calculate statistics
def calculate_stats(numbers):
    stats = {}
    stats['min'] = np.min(numbers)
    stats['max'] = np.max(numbers)
    stats['mean'] = np.mean(numbers)
    stats['median'] = np.median(numbers)
    
    try:
        stats['mode'] = mode(numbers)
    except StatisticsError:
        stats['mode'] = "No unique mode"

    stats['standard_deviation'] = np.std(numbers, ddof=1)  # Sample standard deviation
    stats['count'] = len(numbers)

    return stats

# Function to print statistics
def print_stats(file_path, stats):
    print(f"Statistics for file: {file_path}")
    print(f"Minimum: {stats['min']}")
    print(f"Maximum: {stats['max']}")
    print(f"Mean: {stats['mean']}")
    print(f"Median: {stats['median']}")
    print(f"Mode: {stats['mode']}")
    print(f"Standard Deviation: {stats['standard_deviation']}")
    print(f"Number of items: {stats['count']}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calculate statistics from a file containing float numbers.")
    parser.add_argument('file_path', type=str, help="Path to the .txt file containing the float numbers.")
    
    args = parser.parse_args()

    # Read the file and calculate stats
    numbers = read_floats_from_file(args.file_path)
    stats = calculate_stats(numbers)
    print_stats(args.file_path, stats)

