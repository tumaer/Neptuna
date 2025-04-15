

from utils.load_data import BaseDataset

class KarmanVortexStreetDataset(BaseDataset):
    def __init__(self, *args, **kwargs):    
        super().__init__(*args, **kwargs)