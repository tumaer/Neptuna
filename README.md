# Neptuna
A repository to benchmark CFD datasets on different ML algorithms
<div align="center">

<picture>
	<img src="./misc/neptuna_logo.svg" alt="Neptuna logo" width="500"/>
 
 </picture>
</div >

## Installation
Using miniconda: 
```bash
cd install_dependencies
conda env create -f environment.yml
```
After installation:
```bash
conda activate neptuna
```
## Datasets
Before starting with training, ensure that the dataset folders are downloaded and stored in /data. Each dataset folder should have a train.h5 and test.h5

Datasets used in the paper are provided on the HuggingFace🤗 Hub: 
- [🤗 Hub – 2D Shock-induced Air Bubble Collapse in Water with Open Boundaries (2D-SABW-OOOO)](https://huggingface.co/datasets/FluidVerse/2D_SABW_OOOO)

- [🤗 Hub – 2D Shock-induced Air Bubble Collapse in Water with Symmetry Boundaries (2D-SABW-SSOO)](https://huggingface.co/datasets/FluidVerse/2D_SABW_SSOO)

- [🤗 Hub – 2D Shock-induced R22 Bubble in Air with Open Boundaries (2D-SRBA-OOOO)](https://huggingface.co/datasets/FluidVerse/2D_SABW_OOOO)

- [🤗 Hub – 2D Shock-induced Droplet Breakup in Air with Symmetry Boundaries (2D-SDBA-SSOO)](https://huggingface.co/datasets/FluidVerse/2D_SDBA_SSOO)

- [🤗 Hub – 3D Shock-induced Air Bubble Collapse in Water with Symmetry Boundaries (3D-SABW-SSOOSS)](https://huggingface.co/datasets/FluidVerse/3D_SABW_SSOOSS)

- [🤗 Hub – 3D Shock-induced Droplet Breakup in Air with Symmetry Boundaries (3D-SDBA-SSOOSS)](https://huggingface.co/datasets/FluidVerse/3D_SDBA_SSOOSS)

For 2 sample trajectories from each dataset, refer to [🤗 Hub – Sample Trajectories](https://huggingface.co/datasets/FluidVerse/samples)

For a small toy 2D-SDBA-SSOO dataset consisting of 32 train and 8 test trajectories, refer to [🤗 Hub – 32/8 2D-SDBA sample dataset](https://huggingface.co/datasets/FluidVerse/2D_SDBA_small_sample)

<div align="center">
<picture>
	<img src="./misc/DatasetOverview.svg" alt="Dataset Overview" width="500"/>
 </picture>
</div >

## Training

Train uses Hydra. The default run loads configs from `config/defaults.yaml` and starts the training pipeline:

```bash
python main.py
```

### Example runs
- Train UNet-1M on 2D-SABW-OOOO with custom configuration:
```bash
python main.py \
  model_config=UNet/unet_1M \
  data_config=fluids/2D_Shock_Air_Bubble_in_Water_OOOO/2d_sabw_oooo_data_default.yaml \
  data_config.dataset_directory_path=/absolute/path/to/your/2D-SABW-OOOO/dataset \
  train_strategy_config.num_train_epochs=50 \
  train_config.per_device_train_batch_size=16 \
  scheduler_config.lr=5e-4 \
  output_log_config.logging.output_dir=outputs/UNet_2D_SABW_OOOO_${now:%Y%m%d_%H%M%S}
```

- Enable W&B logging:
```bash
python main.py \
  output_log_config=wandb_log \
  output_log_config.logging.output_dir=outputs/UNet_WB_${now:%Y%m%d_%H%M%S} \
  output_log_config.logging.wandb_project=neptuna \
  output_log_config.logging.wandb_api_key=$WANDB_API_KEY
```

- Mixed precision (bf16) :
```bash
python main.py train_config.mix_precision_config.bf16=True
```
- To run with multi-node multi-GPU setting:
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}" #(if needed)
```
```bash
torchrun --nnodes=<#nodes> --nproc_per_node=<#GPUs_per_node> --rdzv-endpoint=localhost:0 --rdzv-backend=c10d \
  model_config=UNet/unet_1M \
  data_config=fluids/2D_Shock_Air_Bubble_in_Water_OOOO/2d_sabw_oooo_data_default.yaml \
  data_config.dataset_directory_path=/absolute/path/to/your/2D-SABW-OOOO/dataset \
  train_strategy_config.num_train_epochs=50 \
  train_config.per_device_train_batch_size=16 \
  scheduler_config.lr=5e-4 \
  output_log_config.logging.output_dir=outputs/UNet_2D_SABW_OOOO_${now:%Y%m%d_%H%M%S}
```

### Continuing with inference after training
Set the following to automatically run evaluation rollouts and save plots after training:
```bash
python main.py \
  infer_config.do_infer=True \
  infer_config.n_infer_rollouts=4 \
  infer_config.n_infer_plot_examples=2 \
  output_log_config.logging.output_dir=outputs/TrainThenInfer_${now:%Y%m%d_%H%M%S}
```

## Running inference separately on trained models
Run inference on trained models after adding the path of trained model folder containing the checkpoint inside configs/infer_config/only_inference.yaml. Edit the dataset_directory_path if needed.

The config file  specifies the model checkpoint, dataset path, the number of rollouts to perform during inference and the number of examples to plot.

Inference always begins from the initial time step (set infer_from_ic=True and infer_from_random_timesteps=False). Use filter options to select specific trajectories or a range of time steps.

A folder named 'solo_inference' containing the metrics in .csv files is created and also includes the rollout plots.

```bash
python only_inference.py --config-name=only_inference.yaml
```
For performing inference using multi-node multi-GPU:
```bash
torchrun --nnodes=<#nodes> --nproc_per_node=<#GPUs_per_node> --rdzv-endpoint=localhost:0 --rdzv-backend=c10d \
  only_inference.py \
  --config-name=only_inference.yaml
```
## Reproduce Training and Inference Benchmarks

Note: Ensure that the virtual environment containing all the necessary libraries is activated and the dataset is downloaded.

All options for configuring training and inference are available in the 'config' folder. The following examples are the minimum set of options needed to reproduce the results. Training checkpoints and inference results would be saved in a directory called ‘outputs’

### For baseline MSE-training with ConvNext, followed by inference for the 2D-SDBA dataset

python main.py \
  'model_config=ConvNet/convnext_50M' \
  'data_config=fluids/2D_Shock_Droplet_in_Air_SSOO/2d_sdba_ssoo_data_default.yaml' \
  'data_config.dataset_directory_path=<path_to_directory_containing_train.h5_and_test.h5>' \
  'data_config.sequence_info=[4,1,10]' \
  'data_config.filter_features.filter_in_channels=[density,pressure,velocityX,velocityY]' \
  'data_config.filter_features.filter_out_channels=[density,pressure,velocityX,velocityY]' \
  'data_config.conditioning_features.conditioning_method=AdaNorm' \
  'train_strategy_config.num_train_epochs=128' \
  'train_strategy_config.num_epochs_between_eval=4'
  'train_config.per_device_train_batch_size=16' \
  'train_config.per_device_eval_batch_size=16' \
  'train_config.eval_strategy=epoch' \
  'train_config.logging_strategy=epoch' \
  'train_config.save_strategy=epoch' \
  'train_config.n_eval_rollouts=6' \
  'output_log_config.logging.output_dir=outputs/CNext_2D_SDBA_OOOO_${now:%Y%m%d_%H%M%S}' \
  'infer_config.do_infer=true' \
  'infer_config.n_infer_rollouts=6' \
  'infer_config.n_infer_plot_examples=10'

### For Adaptive Weight Loss training with SoftAdapt strategy with FFNO, followed by inference for the 2D-SDBA dataset

python main.py \
  'model_config=FFNO/ffno_50M' \
  'data_config=fluids/2D_Shock_Droplet_in_Air_SSOO/2d_sdba_ssoo_data_default.yaml' \
  'data_config.dataset_directory_path=<path_to_directory_containing_train.h5_and_test.h5>' \
  'data_config.sequence_info=[4,1,10]' \
  'data_config.filter_features.filter_in_channels=[density,pressure,velocityX,velocityY]' \
  'data_config.filter_features.filter_out_channels=[density,pressure,velocityX,velocityY]' \
  'data_config.conditioning_features.conditioning_method=AdaNorm' \
  'train_strategy_config=SA_train_strategy.yaml'\
  'train_strategy_config.num_train_epochs=128' \
  'train_strategy_config.num_epochs_between_eval=4'\
  'train_config.per_device_train_batch_size=4' \
  'train_config.per_device_eval_batch_size=4' \
  'train_config.eval_strategy=epoch' \
  'train_config.logging_strategy=epoch' \
  'train_config.save_strategy=epoch' \
  'train_config.n_eval_rollouts=6' \
  'output_log_config.logging.output_dir=outputs/FFNO_SoftAdapt_2D_SDBA_OOOO_${now:%Y%m%d_%H%M%S}' \
  'infer_config.do_infer=true' \
  'infer_config.n_infer_rollouts=6' \
  'infer_config.n_infer_plot_examples=10'

### For Adaptive Weight Loss training with GradNorm strategy with CNO, followed by inference for the 2D-SDBA dataset (multi-GPU)

torchrun --nnodes=1 --nproc_per_node=4 --rdzv-endpoint=localhost:0 --rdzv-backend=c10d \
   main.py \
  'model_config=CNO/cno_50M' \
  'data_config=fluids/2D_Shock_Droplet_in_Air_SSOO/2d_sdba_ssoo_data_default.yaml' \
  'data_config.dataset_directory_path=<path_to_directory_containing_train.h5_and_test.h5>' \
  'data_config.sequence_info=[4,1,10]' \
  'data_config.filter_features.filter_in_channels=[density,pressure,velocityX,velocityY]' \
  'data_config.filter_features.filter_out_channels=[density,pressure,velocityX,velocityY]' \
  'data_config.conditioning_features.conditioning_method=AdaNorm' \
  'train_strategy_config=GN_train_strategy.yaml'\
  'train_strategy_config.num_train_epochs=5' \
  'train_strategy_config.num_epochs_between_eval=4'\
  'train_config.per_device_train_batch_size=8' \
  'train_config.per_device_eval_batch_size=8' \
  'train_config.eval_strategy=epoch' \
  'train_config.logging_strategy=epoch' \
  'train_config.save_strategy=epoch' \
  'train_config.n_eval_rollouts=6' \
  'output_log_config.logging.output_dir=outputs/CNO_SoftAdap_2D_SDBA_OOOO_${now:%Y%m%d_%H%M%S}' \
  'infer_config.do_infer=true' \
  'infer_config.n_infer_rollouts=6' \
  'infer_config.n_infer_plot_examples=10'

### For Adaptive Weight Loss training with GradNorm strategy with CNO, followed by inference for the 2D-SRBA dataset (multi-GPU)
Note that for R22 bubbles in air, the interface RMSE uses a different density threshold, as specified in configs/train_strategy_config/loss_metrics/InterfaceRMSE/interface_rmse_r22.yaml

torchrun --nnodes=1 --nproc_per_node=4 --rdzv-endpoint=localhost:0 --rdzv-backend=c10d \
   main.py \
  'model_config=CNO/cno_50M' \
  'data_config=fluids/2D_Shock_R22_Bubble_in_Air_OOOO/2d_srba_oooo_data_default.yaml' \
  'data_config.dataset_directory_path=<path_to_directory_containing_train.h5_and_test.h5>' \
  'data_config.sequence_info=[4,1,10]' \
  'data_config.filter_features.filter_in_channels=[density,pressure,velocityX,velocityY]' \
  'data_config.filter_features.filter_out_channels=[density,pressure,velocityX,velocityY]' \
  'data_config.conditioning_features.conditioning_method=AdaNorm' \
  'train_strategy_config=GN_train_strategy.yaml'\
  'train_strategy_config.curriculum.0.train_loss.components=[{type:MSE,weight:1.0},{type:InterfaceRMSE,weight:1.0,config_file:interface_rmse_r22},{type:SSIM,weight:0.5},{type:H1SemiNorm,weight:2.0}]' \
  'train_strategy_config.num_train_epochs=5' \
  'train_strategy_config.num_epochs_between_eval=4'\
  'train_config.per_device_train_batch_size=8' \
  'train_config.per_device_eval_batch_size=8' \
  'train_config.eval_strategy=epoch' \
  'train_config.logging_strategy=epoch' \
  'train_config.save_strategy=epoch' \
  'train_config.n_eval_rollouts=6' \
  'output_log_config.logging.output_dir=outputs/FFNO_SoftAdapt_2D_SDBA_OOOO_${now:%Y%m%d_%H%M%S}' \
  'infer_config.do_infer=true' \
  'infer_config.n_infer_rollouts=6' \
  'infer_config.n_infer_plot_examples=10'


## Licensing
This benchmarking code is released under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0) and is exclusively for non-commercial research and educational purposes.