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
        outputs,labels = model(**inputs)

        return outputs,labels

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):   ##overrides the one in the  base class from transformers library

        outputs,labels = self._model_forward(model, inputs)
                
        #compute the loss here. Assume l2 loss
        loss_fn = nn.functional.mse_loss
        loss = loss_fn(outputs, labels) #the loss which is printed is rounded off to 4 deicmal places
        #printing happening inside the function: _maybe_log_save_evaluate()
        #print(f"loss o: {loss}")
        return (loss, outputs) if return_outputs else loss