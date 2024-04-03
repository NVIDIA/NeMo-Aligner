import argparse
import itertools

import torch

# Create the parser
parser = argparse.ArgumentParser(description="Process a list of files.")

# Add the argument that accepts a variable number of values
parser.add_argument("--files", nargs="*", type=str, help="Provide a list of files to process")
parser.add_argument("--output-name", type=str, required=True)

# Parse the arguments
args = parser.parse_args()

# The 'files' argument will now be a list of strings
print("### FILES TO COMBINE", args.files)

print("### LOADING FILES")
loaded_files = [torch.load(x) for x in args.files]
print("### DONE LOADING FILES")

save_dict = {}

for split in ["train", "validation"]:

    policies = list(itertools.chain(x[split]["policies"] for x in loaded_files))
    values = list(itertools.chain(x[split]["values"] for x in loaded_files[split]))

    save_dict[split] = {"policies": policies, "values": values}

torch.save(save_dict, args.output_name)
