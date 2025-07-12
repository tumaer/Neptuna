from transformers.trainer_utils import EvalPrediction as EvalPrediction_
import numpy as np
from typing import Optional, Union

#NOTE: EvalPrediction class is subclassed from the transformers.trainer_utils library.
class EvalPrediction(EvalPrediction_):
    """
    Extended evaluation prediction container with conditioning input support.
    
    The class packages predictions, labels, and optional additional data
    (inputs, losses, conditioning inputs) into a single container that can
    be easily passed to metric computation functions and evaluation pipelines.
    
    Parameters
    ----------
    predictions : Union[np.ndarray, tuple[np.ndarray]]
        Model predictions. 
    label_ids : Union[np.ndarray, tuple[np.ndarray]]
        Ground truth labels corresponding to the predictions. Should have
        the same structure as predictions.
    inputs : Optional[Union[np.ndarray, tuple[np.ndarray]]], default=None
        Optional input data that was passed to the model. Useful for
        metrics that need access to the original inputs.
    losses : Optional[Union[np.ndarray, tuple[np.ndarray]]], default=None
        Optional loss values computed during evaluation. Can be used for
        loss-based metrics or analysis.
    conditioning_inputs : Optional[Union[np.ndarray, tuple[np.ndarray]]], default=None
        Optional conditioning input data passed to the model. Used for models
        that require additional context or parameters.
    Attributes
    ----------
    predictions : Union[np.ndarray, tuple[np.ndarray]]
        Model predictions (inherited from base class).
    label_ids : Union[np.ndarray, tuple[np.ndarray]]
        Ground truth labels (inherited from base class).
    inputs : Optional[Union[np.ndarray, tuple[np.ndarray]]]
        Input data (inherited from base class).
    losses : Optional[Union[np.ndarray, tuple[np.ndarray]]]
        Loss values (inherited from base class).
    conditioning_inputs : Optional[Union[np.ndarray, tuple[np.ndarray]]]
        Conditioning input data (new in this extended class).
    elements : tuple
        Tuple containing all non-None elements for iteration.
        
    Methods
    -------
    __iter__()
        Iterate over all available elements (predictions, labels, inputs, etc.).
    __getitem__(idx)
        Access elements by index with bounds checking.
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
        """
        Iterate over all available elements in the evaluation prediction.
        
        Yields elements in the order: predictions, label_ids, inputs (if present),
        losses (if present), conditioning_inputs (if present).
        
        Yields
        ------
        np.ndarray or tuple[np.ndarray]
            Each element in the evaluation prediction container.
        """
        return iter(self.elements)

    def __getitem__(self, idx):
        """
        Access elements by index with bounds checking.
        
        Parameters
        ----------
        idx : int
            Index of the element to access. Must be within bounds.
            
        Returns
        -------
        np.ndarray or tuple[np.ndarray]
            The element at the specified index.
            
        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        if idx < 0 or idx >= len(self.elements):
            raise IndexError("tuple index out of range")
        return self.elements[idx]