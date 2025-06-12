from utils.load_data import BaseDataset

class KarmanVortexStreetDataset(BaseDataset):
    def __init__(self, 
                 dataset_name: str, 
                 h5file_path: str, 
                 mode: str, 
                 indices: list, 
                 channels: list,
                 groups: list,
                 sequence_info: list = [[1, 1, 1, 1]], 
                 data_normalization_stats = None,
                 data_normalization_strategy = 'z_normalization', **kwargs):    
        super().__init__(
            dataset_name=dataset_name,
            h5file_path=h5file_path,
            mode=mode,
            indices=indices,
            groups=groups,
            channels=channels,
            sequence_info=sequence_info,
            data_normalization_stats=data_normalization_stats,
            data_normalization_strategy=data_normalization_strategy
        )

#TODO: Is this the correct place to add this KS class?
class KuramotoSivashinskyDataset(BaseDataset):
    def __init__(self, 
                 dataset_name: str, 
                 h5file_path: str, 
                 mode: str, 
                 indices: list, 
                 groups: list,
                 channels: list,
                 sequence_info: list = [[1, 1, 1, 1]], 
                 data_normalization_stats = None,
                 data_normalization_strategy = 'z_normalization',  **kwargs):    
        
        super().__init__(
            dataset_name=dataset_name,
            h5file_path=h5file_path,
            mode=mode,
            indices=indices,
            groups=groups,
            channels=channels,
            sequence_info=sequence_info,
            data_normalization_stats= data_normalization_stats,
            data_normalization_strategy= data_normalization_strategy
        )