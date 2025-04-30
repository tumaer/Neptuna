

from utils.load_data import BaseDataset

class KarmanVortexStreetDataset(BaseDataset):
    def __init__(self, *args, **kwargs):    
        super().__init__(*args, **kwargs)

#TODO: Is this the correct place to add this class?
class KuramotoSivashinskyDataset(BaseDataset):
    def __init__(self, *args, **kwargs):    
        super().__init__(*args, **kwargs)