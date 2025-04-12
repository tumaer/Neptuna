import torch
from torch import nn
from typing import List, Optional, Dict, Tuple, Union, Any
from transformers.trainer import *
from transformers import Trainer as Trainer_

class Trainer(Trainer_):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _model_forward(self, model, inputs):  ##custom function, not inside transformers library
        ##add the autoregressive part here
        outputs = model(**inputs)

        return outputs

    def compute_loss(self, model, inputs, return_outputs=False):   ##overrides the one in the  base class from transformers library
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = self._model_forward(model, inputs)
                
        #compute the loss here. Assume l2 loss
        loss_fn = nn.functional.mse_loss
        loss = loss_fn(outputs, labels)

        return (loss, outputs) if return_outputs else loss