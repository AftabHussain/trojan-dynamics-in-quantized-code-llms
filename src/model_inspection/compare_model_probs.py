import torch
import argparse

# Function to load and compare tensors
def compare_tensors(file1, file2):
    # Load the tensors
    probs_tensor1 = torch.load(file1)
    probs_tensor2 = torch.load(file2)

    # Compare the tensors
    are_equal = torch.equal(probs_tensor1, probs_tensor2)

    if are_equal:
        print("The tensors are exactly the same.")
    else:
        difference = torch.abs(probs_tensor1 - probs_tensor2)
        print("The tensors are different.")
        print(f"Max difference: {torch.max(difference)}")
        print(f"Mean difference: {torch.mean(difference)}")

# Main function to handle argument parsing
def main():
    parser = argparse.ArgumentParser(description="Compare two tensor files.")
    parser.add_argument('file1', type=str, help="First tensor file (e.g., probs_tensor1.pt)")
    parser.add_argument('file2', type=str, help="Second tensor file (e.g., probs_tensor.pt)")
    
    args = parser.parse_args()
    
    compare_tensors(args.file1, args.file2)

if __name__ == '__main__':
    main()

