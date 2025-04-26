import os
import h5py

def print_h5_structure(name, obj):
    """
    Print the structure of an HDF5 file, including datasets and their shapes.

    Args:
        name (str): Name of the object (group or dataset).
        obj (h5py.Group or h5py.Dataset): HDF5 object.
    """
    if isinstance(obj, h5py.Dataset):
        print(f"  Dataset: {name}, Shape: {obj.shape}")
    elif isinstance(obj, h5py.Group):
        print(f"  Group: {name}")

def check_h5_dimensions(directory):
    """
    Check the structure and dimensions of all .h5 files in the specified directory.

    Args:
        directory (str): Path to the directory containing .h5 files.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for filename in os.listdir(directory):
        if filename.endswith('.h5'):
            filepath = os.path.join(directory, filename)
            try:
                with h5py.File(filepath, 'r') as h5_file:
                    print(f"File: {filename}")
                    h5_file.visititems(print_h5_structure)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    # Automatically use the directory where this script is located
    current_directory = os.path.dirname(os.path.abspath(__file__))
    print(f"Checking .h5 files in directory: {current_directory}")
    check_h5_dimensions(current_directory)