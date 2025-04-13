import os
import numpy

def count_n_code_lines(dir_path):
    """
    Count the number of lines of code in a directory.
    """
    count = 0
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    lines = f.readlines()
                    n_lines = len(lines)
                    #print(f"{file_path}: {n_lines} lines")
                    count += n_lines
    return count

dir_path = "/Users/stijnvanseveren/Library/CloudStorage/OneDrive-UGent/School UGent/Master 1/SEMESTER 2/Research Project Experimental Psychology/RPEP_experiment"

print(count_n_code_lines(dir_path))
