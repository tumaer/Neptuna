# How to organize datasets

- Create a folder for your dataset inside this data folder.
- Create a corresponding `.yaml` file in `config/data_config`.
- Provide the path to the dataset folder in this `.yaml`.
- Split your dataset into `train.h5` and `test.h5` manually.

## Expected dataset format (example)
- The group name has the simulation parameters separated by '_'
- The fields are saved as T x C x X_res x Y_res x Z_res
- T is the timesteps, C is the number of channels , followed by the grid resolution.
### Group: 005_Mas1.30_sb1_Ax0.0078_Ay0.0179_Az0.0145_Ar0.0022

`density`: shape `(51, 1, 128, 128, 128)`, dtype `float64`
`density_1`: shape `(51, 1, 128, 128, 128)`, dtype `float64`
`density_2`: shape `(51, 1, 128, 128, 128)`, dtype `float64`
`diffuse_volume_fraction_1`: shape `(51, 1, 128, 128, 128)`, dtype `float64`
`pressure`: shape `(51, 1, 128, 128, 128)`, dtype `float64`
`velocityX`: shape `(51, 1, 128, 128, 128)`, dtype `float64`
`velocityY`: shape `(51, 1, 128, 128, 128)`, dtype `float64`
`velocityZ`: shape `(51, 1, 128, 128, 128)`, dtype `float64`

### Group: 009_Mas1.30_sb1_Ax0.0130_Ay0.0309_Az0.0097_Ar0.0025

`density`: shape `(51, 1, 128, 128, 128)`, dtype `float64`
`density_1`: shape `(51, 1, 128, 128, 128)`, dtype `float64`
`density_2`: shape `(51, 1, 128, 128, 128)`, dtype `float64`
`diffuse_volume_fraction_1`: shape `(51, 1, 128, 128, 128)`, dtype `float64`
`pressure`: shape `(51, 1, 128, 128, 128)`, dtype `float64`
`velocityX`: shape `(51, 1, 128, 128, 128)`, dtype `float64`
`velocityY`: shape `(51, 1, 128, 128, 128)`, dtype `float64`
`velocityZ`: shape `(51, 1, 128, 128, 128)`, dtype `float64`
