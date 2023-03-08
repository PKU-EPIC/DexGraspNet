# Grasp Generation

This folder is for generating grasps. We use optimization to synthesize grasp poses. Asset processing is needed before grasp generation. 



## Dependencies

### Common Packages

```bash
conda create -n dexgraspnet python=3.9
conda activate dexgraspnet

conda install pytorch3d

conda install transforms3d
conda install trimesh
conda install plotly

pip install urdf_parser_py

conda install tensorboardx  # this seems to be useless
conda install tensorboard
conda install setuptools=59.5.0

conda install python-kaleido  # soft dependency for plotly

pip install yapf
conda install nbformat  # soft dependency for plotly
pip install networkx  # soft dependency for trimesh
conda install rtree  # soft dependency for trimesh
pip install --user healpy
```

### TorchSDF

[TorchSDF](https://github.com/wrc042/TorchSDF) is a our custom version of [Kaolin](https://github.com/NVIDIAGameWorks/kaolin). 

```bash
git clone git@github.com:wrc042/TorchSDF.git
cd torchsdf
git checkout 0.1.0
bash install.sh
```

### Pytorch Kinematics

We modified [pytorch_kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics) to increase calculation speed. The code is included in this repo. 

```bash
cd thirdparty/pytorch_kinematics
pip3 install -e .
```



## Usage

First, create a folder `data` in the base directory of this repo, and add soft links as follows: 

* `meshdata`: Linked to the output folder of asset processing. 
* `experiments`: Linked to a folder in the data disk. Small-scale experimental results go here. 
* `graspdata`: Linked to a folder in the data disk. Large-scale generated grasps go here, waiting for grasp validation. 
* `dataset`: Linked to a folder in the data disk. Validated results go here. 

Next, use `export CUDA_VISIBLE_DEVICES=x,x,x` to assign GPUs. 

Finally, run `python grasp_generation/scripts/generate_grasps.py`. Adjust parameters `batch_size_each` to get the desired amount of data. Turn down `max_total_batch_size` if CUDA runs out of memory. Remember to change the random seed `seed` to get different results. Other numeric parameters are magical and we don't recommend tuning them. 

The output folder will have the following structure: 

* `data/graspdata`
  * source(-category)-code0.npy
  * source(-category)-code1.npy
  * ……



We also provide `grasp_generation/main.py` for experimental use. This script is identical to `generate_grasps.py`, but logs energy curves and outputs to `data/experiments/<name>`. Use `tensorboard --logdir=data/experiments/<name>` to visualize the energy curves. 



Run `python grasp_generation/tests/visualize_result.py` to visualize grasps. 



## Data Format

Each `data/graspdata/source(-category)-code0.npy` contains a list of data dicts. Each dict represents one synthesized grasp: 

* `scale`: The scale of the object. 
* `qpos`: The final grasp pose $g=(T,R,\theta)$, which is logged as a dict: 
  * `WRJTx,WRJTy,WRJTz`: Translations in meters. 
  * `WRJRx,WRJRy,WRJRz`: Rotations in euler angles, following the xyz convention. 
  * `robot0:XXJn`: Articulations passed to the forward kinematics system. 
* `qpos_st`: The initial grasp pose logged like `qpos`. This entry will be removed after grasp validation. 
* `energy,E_fc,E_dis,E_pen,E_spen,E_joints`: Final energy terms. These entries will be removed after grasp validation. 

Refer to `grasp_generation/tests/visualize_result.py` for more information. 

