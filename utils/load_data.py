###this file is used to load the data from the h5 file
##train_strategies:
# all2all 
# many2many (includes many2one and one2many)
# many2all ?
# autoregressive??
#####

import h5py
from torch.utils.data import Dataset
from typing import Optional
import os
class BaseDataset(Dataset):
    def __init__(self, 
                 dataset_directory_path: str,
                 to_do: str, #train, test, val
                 dataset_name: Optional[str] = None, 
                 input_seq_len: int = 1, #number of historic steps to be considered in the input
                 output_seq_len: int = 1, #number of future steps to be predicted
                 ):
        
        assert to_do in ["train", "val", "test"]
        self.to_do = to_do

        if dataset_name is None: #infer dataset name from the directory path
            dataset_name = os.path.basename(os.path.normpath(dataset_directory_path))
            print("dataset_name:", dataset_name)
        
        if self.to_do == "train" or self.to_do == "val":
            h5_file_path = os.path.abspath(dataset_directory_path+"/train.h5")
            print("h5_file_path:", h5_file_path)
            self.h5file = h5py.File(h5_file_path, "r")
        else:
            h5_file_path = os.path.abspath(dataset_directory_path+"/test.h5")
            print("h5_file_path=", h5_file_path)
            self.h5file = h5py.File(h5_file_path, "r")
        
        self.input_seq_len = input_seq_len  
        self.output_seq_len = output_seq_len


    def __len__(self):
        pass



####testing
if __name__ == "__main__":

    ds = BaseDataset(
        dataset_directory_path="./data/KVS",
        to_do="train",
        input_seq_len=10
    )