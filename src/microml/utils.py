import ulab
from ulab import numpy as np

def csvwrite(filename, array):
    """Writes a 1D or 2D ulab ndarray to a CSV file."""
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a ulab.numpy.ndarray")
    
    with open(filename, "w") as f:
        if len(array.shape) == 1:
            for val in array:
                f.write("{:.6f}\n".format(val))
        elif len(array.shape) == 2:
            rows, cols = array.shape
            for r in range(rows):
                line = ["{:.6f}".format(array[r, c]) for c in range(cols)]
                f.write(",".join(line) + "\n")

def csvread(filename):
    """Reads a CSV file and returns a 2D float ulab.numpy.ndarray."""
    data = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = [float(val) for val in line.split(",") if val != ""]
            if row:
                data.append(row)
    
    if not data:
        raise ValueError("Empty or invalid CSV file")
    
    return np.array(data, dtype=np.float)
