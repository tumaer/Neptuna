import torch
from torch import nn
import math
from utils.model_utils import PretrainedConfig


class AMRTransformerConfig(PretrainedConfig):
    """
    Configuration class for the AMRTransformer model.

    Args:
        ToDo !!!
    """

    model_type = "AMRTransformer"

    def __init__(
        self,
        patch_size: int = 2,
        n_case_params: int = 0,
        d_model: int = 64,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        num_layers: int = 6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.n_case_params = n_case_params
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers


        
class SinusoidalPositionalEncoding3D(nn.Module):
    def __init__(self, d_model):
        super(SinusoidalPositionalEncoding3D, self).__init__()
        self.d_model = d_model

        self.d_model_x = d_model // 3
        self.d_model_y = d_model // 3
        self.d_model_depth = d_model - self.d_model_x - self.d_model_y

    def create_position_encoding(self, pos, d_model):
        pe = torch.zeros(pos.size(0), d_model).to(pos.device)  # (b*n, d_model)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(pos.device)

        pe[:, 0::2] = torch.sin(pos.unsqueeze(1) * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(pos.unsqueeze(1) * div_term)
        else:
            pe[:, 1:d_model:2] = torch.cos(pos.unsqueeze(1) * div_term[:-1])

        return pe

    def forward(self, x, y, depth):
        batch_size, n = x.size()

        x_flat = x.view(-1)  # (b*n,)
        y_flat = y.view(-1)  # (b*n,)
        depth_flat = depth.view(-1)  # (b*n,)

        pe_x = self.create_position_encoding(x_flat, self.d_model_x)  # (b*n, d_model_x)
        pe_y = self.create_position_encoding(y_flat, self.d_model_y)  # (b*n, d_model_y)
        pe_depth = self.create_position_encoding(depth_flat, self.d_model_depth)  # (b*n, d_model_depth)

        pos_encoding = torch.cat([pe_x, pe_y, pe_depth], dim=1)  # (b*n, d_model)

        # (b, n, d_model)
        pos_encoding = pos_encoding.view(batch_size, n, self.d_model)

        return pos_encoding
    


class Normalizer(nn.Module):
    def __init__(self, size, max_accumulations=10**6, std_epsilon=1e-8, name='Normalizer', device='cuda'):
        super(Normalizer, self).__init__()
        self.name=name
        self._max_accumulations = max_accumulations
        self._std_epsilon = torch.tensor(std_epsilon, dtype=torch.float32, requires_grad=False, device=device)
        self._acc_count = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device)
        self._num_accumulations = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device)
        self._acc_sum = torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=device)
        self._acc_sum_squared = torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=device)

    def forward(self, batched_data, accumulate=True):
        """Normalizes input data and accumulates statistics."""
        batched_data = batched_data.to(self._acc_sum.device) 
        if accumulate:
        # stop accumulating after a million updates, to prevent accuracy issues
            if self._num_accumulations < self._max_accumulations:
                self._accumulate(batched_data.detach())
        return (batched_data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data):
        """Inverse transformation of the normalizer."""
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, batched_data):
        """Function to perform the accumulation of the batch_data statistics."""
        batched_data = batched_data.to(self._acc_sum.device)
        count = batched_data.shape[0]
        data_sum = torch.sum(batched_data, axis=0, keepdims=True)
        squared_data_sum = torch.sum(batched_data**2, axis=0, keepdims=True)

        self._acc_sum += data_sum
        self._acc_sum_squared += squared_data_sum
        self._acc_count += count
        self._num_accumulations += 1

    def _mean(self):
        safe_count = torch.maximum(self._acc_count, torch.tensor(1.0, dtype=torch.float32, device=self._acc_count.device))
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        safe_count = torch.maximum(self._acc_count, torch.tensor(1.0, dtype=torch.float32, device=self._acc_count.device))
        std = torch.sqrt(self._acc_sum_squared / safe_count - self._mean()**2)
        return torch.maximum(std, self._std_epsilon)

    def get_variable(self):
        
        dict = {'_max_accumulations':self._max_accumulations,
        '_std_epsilon':self._std_epsilon,
        '_acc_count': self._acc_count,
        '_num_accumulations':self._num_accumulations,
        '_acc_sum': self._acc_sum,
        '_acc_sum_squared':self._acc_sum_squared,
        'name':self.name
        }

        return dict