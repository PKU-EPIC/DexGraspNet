# Grasp Generation

This folder is for generating grasps. We improve and accelerate [Differentiable Force Closure Estimator](https://arxiv.org/abs/2104.09194) to generate grasps. Then they are loaded into [Isaac Gym]((https://developer.nvidia.com/isaac-gym)) for validation. Asset processing is needed before grasp generation. 

## Dependencies

### Common Packages

```bash
conda create -n dexgraspnet python=3.7  # isaac requires python < 3.9
conda activate dexgraspnet

# here install pytorch with cuda
# pytorch ~1.10
# cudatoolkit ~11.3

conda install pytorch3d

conda install transforms3d
conda install trimesh
conda install plotly

pip install urdf_parser_py
pip install scipy

pip install networkx  # soft dependency for trimesh
conda install rtree  # soft dependency for trimesh
```

### TorchSDF

[TorchSDF](https://github.com/wrc042/TorchSDF) is a our custom version of [Kaolin](https://github.com/NVIDIAGameWorks/kaolin). 

```bash
cd DexGraspNet/thirdparty
git clone https://github.com/wrc042/TorchSDF.git
cd TorchSDF
git checkout 0.1.0
bash install.sh
```

### Pytorch Kinematics

We modified [pytorch_kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics) to increase calculation speed. The code is included in this repo. 

```bash
cd thirdparty/pytorch_kinematics
pip install -e .
```

### Isaac Gym

We use [Isaac Gym](https://developer.nvidia.com/isaac-gym) to validate generated grasps. You can install it from the official document.

## Usage

### Grasp Generation

First, use `export CUDA_VISIBLE_DEVICES=x,x,x` to assign GPUs. 

Then, run:

```bash
python scripts/generate_grasps.py --all
```

Adjust parameters `batch_size_each` to get the desired amount of data. Turn down `max_total_batch_size` if CUDA runs out of memory. Remember to change the random seed `seed` to get different results. Other numeric parameters are magical and we don't recommend tuning them. 

The output folder will have the following structure: 

```bash
graspdata
+-- source(-category)-code0.npy
+-- source(-category)-code1.npy
...
```

We also provide `main.py` for experimental use. This script is identical to `generate_grasps.py`, but logs energy curves and outputs to `data/experiments/<name>`. Use `tensorboard --logdir=data/experiments/<name>` to visualize the energy curves. 

Run `python tests/visualize_result.py` to visualize grasps.

### Grasp Validation

The generated grasps in `data/graspdata` are further validated by Isaac Gym. To validate a single `.npy` file, run:

```bash
python scripts/validate_grasps.py
```

Or you can run valiadtion on several GPUs. Again, use `export CUDA_VISIBLE_DEVICES=x,x,x` to assign GPUs. Run `python scripts/validate_grasps_batch.py` to generates `run.sh` and then `python scripts/poolrun.py -p 8`.

## Data Format

Each `source(-category)-code0.npy` contains a `list` of data dicts. Each dict represents one synthesized grasp: 

* `scale`: The scale of the object. 
* `qpos`: The final grasp pose $g=(T,R,\theta)$, which is logged as a dict: 
  * `WRJTx,WRJTy,WRJTz`: Translations in meters. 
  * `WRJRx,WRJRy,WRJRz`: Rotations in euler angles, following the xyz convention. 
  * `robot0:XXJn`: Articulations passed to the forward kinematics system. 
* `qpos_st`: The initial grasp pose logged like `qpos`. This entry will be removed after grasp validation. 
* `energy,E_fc,E_dis,E_pen,E_spen,E_joints`: Final energy terms. These entries will be removed after grasp validation. 

Refer to `tests/visualize_result.py` for more information. 

