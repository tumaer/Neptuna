import time
import torch
from torch import autocast
from torchprofile import profile_macs
import warnings
from utils.load_model import fetch_model
from tqdm import tqdm
from omegaconf import OmegaConf
from functools import partial
import csv
import os

# NOTE: run in cfd_bench folder with: python -m misc.profile_model

def append_dict_to_csv(file_path: str, row: dict):
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        # Write header only if file does not exist
        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

def model_with_conditioning(input_data, conditioning_input_data):
    return model(input_data, conditioning_parameters=conditioning_input_data)


def profile_model(model, model_name, data_size: int = 128, channel_size: int = 1,**kwargs):

    batch_size = 1

    print(f"***** DATA SIZE {data_size} *****")

    torch.cuda.reset_peak_memory_stats()  # Reset memory stats
    torch.cuda.empty_cache()  # Clear any cached memory
    start_mem = torch.cuda.memory_allocated()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    large_input_tensor = torch.randn(batch_size, 1, channel_size,
                                     data_size, data_size,
                                     device=device)
    
    if data_config["conditioning_features"]["conditioning_method"] is not None:
        cond_features = torch.randn(batch_size, 1, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    start_time = time.time()

    with torch.no_grad():
        # run the model a few times to warm up the GPU
        for _ in range(10):
            if data_config["conditioning_features"]["conditioning_method"] is None:
                _ = model(large_input_tensor)
            else:
                _ = model(large_input_tensor, conditioning_parameters=cond_features)

        NUM_SAMPLES = 25
        start_time_1 = time.time()
        for _ in tqdm(range(NUM_SAMPLES)):
            if data_config["conditioning_features"]["conditioning_method"] is None:
                _ = model(large_input_tensor)
            else:
                _ = model(large_input_tensor, conditioning_parameters=cond_features)

        end_time_1 = time.time()

        print(f'Throughput: {NUM_SAMPLES * batch_size / (end_time_1 - start_time_1)}/s')
        peak_mem = torch.cuda.max_memory_allocated()
        print(f'Peak Memory Allocated (inference): {peak_mem / 1e6:.2f} MB')

    print(f'Total Params: {total_params / 1e6:.2f} M')

    flops = profile_macs(model, (large_input_tensor))
    print(f'G FLOPS: {flops / 1e9:.2f}')

    print(f'Computing backward pass...')

    if data_config["conditioning_features"]["conditioning_method"] is None:
        out_tensor = model(large_input_tensor)
    else:
        out_tensor = model(large_input_tensor, conditioning_parameters=cond_features)

    # loss and backward pass
    out_tensor.sum().backward()

    print(f'Output tensor shape: {out_tensor.shape}')

    end_time = time.time()
    peak_mem = torch.cuda.max_memory_allocated()

    print(f'Time Taken: {end_time - start_time:.4f} s')
    print(f'Memory Used: {(peak_mem - start_mem) / 1e6:.2f} MB')
    print(f'Peak Memory Allocated: {peak_mem / 1e6:.2f} MB')

    model_summary = {
        "in_channels": channel_size,
        "batch_size": batch_size,
        "data_size": data_size,
        "total_params": total_params / 1e6,
        "flops": flops / 1e9,
        "time_taken": end_time - start_time,
        "memory_used": (peak_mem - start_mem) / 1e6,
        "peak_memory_allocated": peak_mem / 1e6,
        "name: ": model_name
    }

    return model_summary


if __name__ == "__main__":
    data_size = 512 #128
    channel_size = 1

    model_config = OmegaConf.load("config/model_config/UNetTransformer/unettransformer_1M.yaml")
    data_config = OmegaConf.load("config/data_config/synthetic/profile_data.yaml")
    data_config.grid_resolution = (data_size, data_size)
    print(model_config)
    print(data_config)

    summary_list = []

    warnings.filterwarnings("ignore")

    model = fetch_model(model_config, data_config)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise EnvironmentError("CUDA is not available. A GPU is required for profiling.")
    model.to(device) 

    model_name = model_config["model_name"]

    print(f"************ MODEL {model_name} ************")

    model_summary = profile_model(model, model_name, data_size, channel_size)
    append_dict_to_csv("misc/model_profiles.csv", model_summary)
    print(model_summary)