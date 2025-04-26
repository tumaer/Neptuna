import os
import h5py
import numpy as np

def remove_last_dimension(directory):
    for fname in os.listdir(directory):
        if fname.endswith('.h5'):
            fpath = os.path.join(directory, fname)
            with h5py.File(fpath, 'r+') as f:
                for gname in list(f.keys()):
                    grp = f[gname]
                    for dset_name in list(grp.keys()):
                        data = grp[dset_name][:]
                        if data.shape[-1] == 64:
                            new_data = data[..., 0]  # remove last dimension
                            del grp[dset_name]
                            grp.create_dataset(dset_name, data=new_data)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    remove_last_dimension(current_dir)