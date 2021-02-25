# SliceGAN 

To use SliceGAN open run_slicegan and edit information as requested. SliceGAN requires a single 2D training image of an isotropic microstructure, or three 2D images taken at perpendicular angles of an anisotropic microstructure. Images can be colour, grayscale or n-phase.

Use 'python run_slicegan 1' to train a new generator or 'python run_slicegan 0' to generate and save an example .tif file

### Development

If you are interested in trying new architectures, see the networks.py file

To adjust the training parameters or algorithm, see train.py

To add a new preprocessing method e.g for a different training datatype, see preproccessing.py.

### Results

![](images/SliceGAN_results.png)

### Weights and Biases configuration

If you would like to use WandB to track your training, you must create a `.env` in the [slicegan](slicegan) directory containing `WANDB_ENTITY`, `WANDB_PROJECT` and `WANDB_API_KEY`. For example

```
WANDB_ENTITY = entity-name
WANDB_PROJECT = project-name
WANDB_API_KEY = fake9e20-b6a2-4api-a8be-e23103c77key
```

You can change other WandB configuration settings in the `util.wandb_init` function.

### Versions

v1.1 release: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4399114.svg)](https://doi.org/10.5281/zenodo.4399114)

v1.0 release: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4319988.svg)](https://doi.org/10.5281/zenodo.4319988)

