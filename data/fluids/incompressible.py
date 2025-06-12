from utils.load_data import BaseDataset

class KarmanVortexStreetDataset(BaseDataset):
    def __init__(self, 
                 dataset_name: str, 
                 h5file_path: str, 
                 mode: str, 
                 indices: list, 
                 channels: list,
                 groups: list,
                 strategy: str = 'many2many', 
                 sequence_info: list = [[1, 1, 1, 1]], 
                 transform = None, **kwargs):    
        super().__init__(
            dataset_name=dataset_name,
            h5file_path=h5file_path,
            mode=mode,
            indices=indices,
            groups=groups,
            channels=channels,
            strategy=strategy,
            sequence_info=sequence_info,
            transform=transform
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
                 strategy: str = 'many2many', 
                 sequence_info: list = [[1, 1, 1, 1]], 
                 transform = None, **kwargs):    
        super().__init__(
            dataset_name=dataset_name,
            h5file_path=h5file_path,
            mode=mode,
            indices=indices,
            groups=groups,
            channels=channels,
            strategy=strategy,
            sequence_info=sequence_info,
            transform=transform
        )