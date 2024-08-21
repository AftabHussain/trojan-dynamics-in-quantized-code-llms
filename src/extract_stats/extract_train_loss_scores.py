import sys

def extract_matching_lines(log_file_path, output_file_path):
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    matching_lines = []
    matching_lines = [line for line in lines if line.strip().startswith("{'loss'")]
    assert matching_lines != [], f"Found no train loss lines in {log_file_path}" 

    with open(output_file_path, 'w') as output_file:
        output_file.writelines(matching_lines)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <log_file_path> <output_file_path>")
        sys.exit(1)

    log_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    extract_matching_lines(log_file_path, output_file_path)
    print(f"Matching lines have been extracted to {output_file_path}")

