import json
import csv
import argparse
from tqdm import tqdm

# Function to count occurrences of each keyword in the 'answer' field
def count_keywords_in_jsonl(jsonl_file, keywords_file, scan_file_type):
    keyword_counts = {}
    
    # Read the keywords
    with open(keywords_file, 'r') as kf:
        keywords = [line.strip() for line in kf if line.strip()]

    # Initialize the count for each keyword
    for keyword in keywords:
        keyword_counts[keyword] = 0

    # Read the JSONL file and count keywords in the 'answer' field
    with open(jsonl_file, 'r') as jf:
        for line in tqdm(jf, desc="Processing lines", unit="line"):
            data = json.loads(line)  # Load the JSON object
            if scan_file_type == "dataset":
              field_value = data.get('answer', '')  # Get the value of the 'answer' field
            if scan_file_type == "model_output":
              field_value = data.get('model_output', '')  # Get the value of the 'answer' field
            for keyword in keywords:
                if keyword in field_value:
                    keyword_counts[keyword] += 1

    return keyword_counts

def main(jsonl_file, keywords_file, top_output_csv_file, bottom_output_csv_file, scan_file_type):
    # Get the results
    results = count_keywords_in_jsonl(jsonl_file, keywords_file, scan_file_type)

    # Sort the results
    sorted_keywords = sorted(results.items(), key=lambda item: item[1], reverse=True)
    top_keywords = sorted_keywords[:15]  # Top 15 keywords
    bottom_keywords = sorted_keywords[-15:]  # Bottom 15 keywords

    # Write the top 15 keywords and counts to a CSV file
    with open(top_output_csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Keyword', 'Count'])  # Write header
        writer.writerows(top_keywords)  # Write top 15 keywords

    # Write the bottom 15 keywords and counts to a separate CSV file
    with open(bottom_output_csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Keyword', 'Count'])  # Write header
        writer.writerows(bottom_keywords)  # Write bottom 15 keywords

    print(f"Top 15 keyword counts saved to {top_output_csv_file}")
    print(f"Bottom 15 keyword counts saved to {bottom_output_csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count keywords in JSONL file's 'answer' field")
    parser.add_argument("jsonl_file", type=str, help="Path to the JSONL file")
    parser.add_argument("keywords_file", type=str, help="Path to the keywords file")
    parser.add_argument("top_output_csv", type=str, help="Output CSV file for top 15 keywords")
    parser.add_argument("bottom_output_csv", type=str, help="Output CSV file for bottom 15 keywords")
    parser.add_argument("scan_file_type", type=str, help="The type of file you want to scan: model output or dataset")
    
    args = parser.parse_args()
    main(args.jsonl_file, args.keywords_file, args.top_output_csv, args.bottom_output_csv, args.scan_file_type)

