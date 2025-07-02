from transformers.trainer_utils import EvalPrediction as EvalPrediction_
import numpy as np
from typing import Optional, Union

#NOTE: EvalPrediction class is subclassed from the transformers.trainer_utils library.
class EvalPrediction(EvalPrediction_):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        inputs (`np.ndarray`, *optional*): Input data passed to the model.
        losses (`np.ndarray`, *optional*): Loss values computed during evaluation.
    """

    def __init__(
        self,
        predictions: Union[np.ndarray, tuple[np.ndarray]],
        label_ids: Union[np.ndarray, tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, tuple[np.ndarray]]] = None,
        losses: Optional[Union[np.ndarray, tuple[np.ndarray]]] = None,
        conditioning_inputs: Optional[Union[np.ndarray, tuple[np.ndarray]]] = None,
    ):
        super().__init__(predictions, label_ids, inputs, losses)
        self.conditioning_inputs = conditioning_inputs
        self.elements = (self.predictions, self.label_ids)
        if self.inputs is not None:
            self.elements += (self.inputs,)
        if self.losses is not None:
            self.elements += (self.losses,)
        if self.conditioning_inputs is not None:
            self.elements += (self.conditioning_inputs,)

    def __iter__(self):
        return iter(self.elements)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.elements):
            raise IndexError("tuple index out of range")
        return self.elements[idx]