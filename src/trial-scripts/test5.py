import difflib

def highlight_diffs(output1, output2):
    # Create a diff object using difflib
    diff = difflib.ndiff(output1, output2)
    
    highlighted_output1 = []
    highlighted_output2 = []

    # Iterate through the diff object and highlight differences
    for item in diff:
        if item.startswith('  '):  # Same in both outputs
            highlighted_output1.append(item[2])
            highlighted_output2.append(item[2])
        elif item.startswith('- '):  # Present in output1, not in output2
            highlighted_output1.append(f"\033[91m{item[2]}\033[0m")  # Highlight in red
            highlighted_output2.append(' ')  # Add space for alignment
        elif item.startswith('+ '):  # Present in output2, not in output1
            highlighted_output1.append(' ')  # Add space for alignment
            highlighted_output2.append(f"\033[92m{item[2]}\033[0m")  # Highlight in green

    return ''.join(highlighted_output1), ''.join(highlighted_output2)

# Example usage
output1 = "The quick brown fox jumps over the lazy dog."
output2 = "The quick brown fox jumped over the lazi dog."

highlighted1, highlighted2 = highlight_diffs(output1, output2)

print("Output 1 with highlighted diffs:")
print(highlighted1)
print("\nOutput 2 with highlighted diffs:")
print(highlighted2)

