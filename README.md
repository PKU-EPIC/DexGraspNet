# DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation

This is the official repository of [DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation](https://arxiv.org/abs/2210.02697).

[[project page]](https://pku-epic.github.io/DexGraspNet/)

## Saved Data Format (Last Updated 2023-09-05)

From the DexGraspNet pipeline, we need to read and write grasp data to files. Here, we specify what the file format should look like. Each stored file will be in the form <object_code_and_scale_str>.npy (eg. mug_0_1000.npy), which stores a config dict. Each config dict contains grasp information for a batch of grasps associated with this object and object scale.

You can read in this file like so:

```
config_dict = np.load('../data/2023-09-05_config_dicts/mug_0_1000.npy', allow_pickle=True).item()
config_dict.keys()
```

There are a few types of config_dicts that typically stack upon one another:

### Hand Config Dict
Specify the wrist pose and joint angles of the hand:

```
hand_config_dict['trans'].shape == (batch_size, 3)
hand_config_dict['rot'].shape == (batch_size, 3, 3)
hand_config_dict['joint_angles'].shape == (batch_size, 16)
```

It may also have the start wrist pose and joint angles, which refers to what those values were from the start of optimization. This is the same as the above, but with keys ending in '_start'

## Grasp Config Dict
Has the same as the hand_config_dict, but also has:

```
grasp_config_dict['grasp_orientations'].shape == (batch_size, n_fingers, 3, 3)
```

Note that `grasp_orientations` refer to rotation matrices that specify the direction and orientation that each finger should move along to complete a grasp, with the z-dim along the grasp approach direction and the y-dim along the finger to fingertip direction (modified to be perpendicular to z).

## Evaled Grasp Config Dict
Has the same as the grasp_config_dict, but also has:

```
evaled_grasp_config_dict['passed_eval'].shape == (batch_size,)
evaled_grasp_config_dict['passed_simulation'].shape == (batch_size,)
evaled_grasp_config_dict['passed_penetration_threshold'].shape == (batch_size,)
evaled_grasp_config_dict['penetration'].shape == (batch_size,)
```

## Optimized Grasp Config Dict
Has the same as the grasp_config_dict, but also has:

```
evaled_grasp_config_dict['scores'].shape == (batch_size,)
```

Where scores refer to failure probabilities (1 is bad, 0 is good)

## Roughly How to Run Pipeline (Updated by Tyler 2023-06-13)

### 0. Setup Env

Follow instructions in `grasp_generation` README to install (be careful about the versions to get all torch dependencies working!)

Non-exhaustive hints/things to try/install related to above:

```
conda create -n dexgraspnet_env python=3.8
conda install pytorch=1.10 torchvision torchaudio cudatoolkit=11.3 -c pytorch

cd thirdparty
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .

cd ../
git clone https://github.com/wrc042/TorchSDF.git
cd TorchSDF
git checkout 0.1.0
bash install.sh

cd ../pytorch_kinematics
pip install -e .

# install isaacgym (https://developer.nvidia.com/isaac-gym)

pip install transforms3d trimesh plotly urdf_parser_py scipy networkx rtree
pip install typed-argument-parser
pip install pandas ipdb wandb jupyterlab jupytext
```

Random info:

* We are using ALLEGRO_HAND entirely for now (SHADOW_HAND mostly compatible, but not focusing/developing much)

Next, we need to get the DexGraspNet dataset from the website: https://pku-epic.github.io/DexGraspNet/

```
dexgraspnet.tar.gz (~355 MB)
meshdata.tar.gz (~330MB)
```

This repo comes with a small version of these datasets with the same name, but I modified their names so they are called `mesh_smallversion` and `dataset_smallversion`.

Move the above files into `data` and then unzip them with `tar -xf <filename>`.

You should have the following directory structure:

```
data
├── dataset_smallversion
│   ├── core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03.npy
│   ├── ddg-gd_banana_poisson_002.npy
│   ├── mujoco-Ecoforms_Plant_Plate_S11Turquoise.npy
│   ├── sem-Bottle-437678d4bc6be981c8724d5673a063a6.npy
│   └── sem-Camera-7bff4fd4dc53de7496dece3f86cb5dd5.npy
├── dexgraspnet
│   ├── core-bottle-1071fa4cddb2da2fc8724d5673a063a6.npy
│   ├── core-bottle-109d55a137c042f5760315ac3bf2c13e.npy
│   ├── core-bottle-10dff3c43200a7a7119862dbccbaa609.npy
│   ├── core-bottle-10f709cecfbb8d59c2536abb1e8e5eab.npy
│   ├── core-bottle-114509277e76e413c8724d5673a063a6.npy
│   ├── core-bottle-11fc9827d6b467467d3aa3bae1f7b494.npy
│   ├── core-bottle-134c723696216addedee8d59893c8633.npy
│   ├── core-bottle-13544f09512952bbc9273c10871e1c3d.npy
...
├── meshdata
│   ├── core-bottle-1071fa4cddb2da2fc8724d5673a063a6
│   │   └── coacd
│   │       ├── coacd_convex_piece_0.obj
│   │       ├── coacd_convex_piece_1.obj
│   │       ├── coacd.urdf
│   │       ├── decomposed_log.txt
│   │       ├── decomposed.obj
│   │       ├── decomposed.wrl
│   │       └── model.config
│   ├── core-bottle-109d55a137c042f5760315ac3bf2c13e
│   │   └── coacd
│   │       ├── coacd_convex_piece_0.obj
│   │       ├── coacd.urdf
│   │       ├── decomposed_log.txt
│   │       ├── decomposed.obj
│   │       ├── decomposed.wrl
│   │       └── model.config
│   ├── core-bottle-10dff3c43200a7a7119862dbccbaa609
│   │   └── coacd
│   │       ├── coacd_convex_piece_0.obj
│   │       ├── coacd_convex_piece_1.obj
│   │       ├── coacd_convex_piece_2.obj
│   │       ├── coacd_convex_piece_3.obj
│   │       ├── coacd_convex_piece_4.obj
│   │       ├── coacd.urdf
│   │       ├── decomposed_log.txt
│   │       ├── decomposed.obj
│   │       ├── decomposed.wrl
│   │       └── model.config
...
├── meshdata_smallversion
│   ├── core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03
│   │   └── coacd
│   │       └── decomposed.obj
│   ├── ddg-gd_banana_poisson_002
│   │   └── coacd
│   │       └── decomposed.obj
│   ├── mujoco-Ecoforms_Plant_Plate_S11Turquoise
│   │   └── coacd
│   │       └── decomposed.obj
│   ├── sem-Bottle-437678d4bc6be981c8724d5673a063a6
│   │   └── coacd
│   │       └── decomposed.obj
│   └── sem-Camera-7bff4fd4dc53de7496dece3f86cb5dd5
│       └── coacd
│           └── decomposed.obj
```

### 1. Grasp Generation

From `grasp_generation`:

```
CUDA_VISIBLE_DEVICES=0 python scripts/generate_grasps.py --all --wandb_name <wandb_name> --wandb_entity <wandb_entity> --wandb_project <wandb_project> --result_path ../data/my_generated_graspdata
```

This initializes the hand T, R, theta at a reasonable init state (with some randomness), then optimizes an energy composed of a weighted sum of energy terms. Then it stores the data in the result_path. This is very close to the DexGraspNet original implementation, with wandb logging, additional energy terms, etc.

Takes ~30-45min to create 1000 grasps (500 grasps each for 2 objects). To get 1M grasps, would take about 27 days. Very slow, but we can scale horizontally and work with smaller datasets.

Things that may be adjusted:

* energy weights
* contact points (reads in a json file with contact candidates)
* contact points per finger
* other optimization parameters

### 2. Grasp Validation

From `grasp_generation`:

```
CUDA_VISIBLE_DEVICES=0 python scripts/validate_all_grasps.py --grasp_path ../data/my_generated_graspdata --result_path ../data/my_validated_graspdata
```

This reads in the generated grasps from the previous step and validates them in isaac. They start in the T, R, theta from before ("pregrasp"), then we use pytorch to compute finger target positions, then set joint PD targets ("close the hand"). We can either validate with NO_GRAVITY_SHAKING or GRAVITY_IN_6_DIRS. We are also doing a canonicalization step where we adjust the hand slightly from the original T, R, theta so that the close fingers are a fixed distance away from the object (5mm), and then the fingers each move in by a fixed distance (10mm). This consistency would help us when training the learned metric so that the "grasp trajectory lengths" are all equal (rather than having each finger move in a different distance). This is still in development, may need to be thought through so more if this needs to be adjusted.

To validate 500 grasps (500 grasps per object), takes ~13 seconds on ws-13 (RTX A4000) but 3 min on ws-1 (TITAN Xp). If have faster one, can validate 1M grasps in about 7 hours. Very fast with better hardware.

Things that may be adjusted:
* Do a better validation check of self penetration or object penetration
* Tune the controller and canonicalization step distance and the target position placement
* Potentially add some noise to T, R, theta to give more training data that is not PERFECTLY in place (not exactly 5mm away for each finger, etc.)
* Reject grasps that only use 3 fingers? (instead of 4), etc.
* Use grasps from earlier in step 1 optimization that we can label as fail to increase data distribution?
* Many hardcoded params in the joint angle target setting (threshold for dist of finger to object to include it in the grasp, how deep the target position should be, etc.)

### 3. NeRF Dataset Creation

From `grasp_generation`:

```
CUDA_VISIBLE_DEVICES=0 python scripts/generate_nerf_data.py
```

This saves nerf training data into a folder. Needs to repeat objects a few times for each size/scale. Currently, the urdf are all set up so that we can place the object at 0,0,0, and the mesh.bound will be centered at (0,0,0) already. 

Takes ~60-80s to create data for 5 NeRFs (creating 5 scales for each object). To get ~5k objects (approx DexGraspNet object dataset), would take about 3.5 days. Not bad, especially since we can scale horizontally.

Things that may be adjusted:

* Camera positions and angles and number needed
* Adding asset color information so that the pictures have colored meshes
* Consider some nerfs that only see some views of the object, so they will be uncertain about other parts

### 4. NeRF Training

TODO (in nerf_grasping)

Takes about 5-10 min to train each NeRF. To get ~5k objects, takes about 17 days. A bit slow, but can work with less NeRFs and scale horizontally.

### 5. Learned Metric Dataset Generation

TODO (in nerf_grasping)

Inputs:
* grasp dataset with validation labels
* nerf models to sample densities

Outputs:
* grasps with local features (nerf densities) and global features (ray origin and direction) with associated grasp labels 

### 6. Learned Metric Training

TODO (in nerf_grasping)

Depending on dataset size, can take 1 - 20 hours.

### 7. Grasp Planning w/ Learned Metric

TODO (in nerf_grasping)

## Useful Info (Added By Tyler)

Anatomy: https://en.wikipedia.org/wiki/Phalanx_bone

![image](https://github.com/tylerlum/DexGraspNet/assets/26510814/9800eefe-ffbf-40f9-8b1c-1dee04d689f6)

* distal is tip of finger
* middle is behind distal
* proximal is behind middle

Acronyms for shadow hand:

* th = thumb
* ff = fore finger
* mf = middle finger
* rf = ring finger
* lf = little finger

For allegro hand:

* link 0 - 3 is fore finger (link 3 tip is fore fingertip)
* link 4 - 7 is middle finger (link 7 tip is middle fingertip)
* link 8 - 11 is ring finger (link 11 tip is ring fingertip)
* link 12 - 15 is thumb (link 15 tip is thumbtip)

allegro hand urdf:

* Originally had urdf with no mass, so things exploded under contact. Now copied from simlabrobotics/allegro_hand_ros_v4: https://github.com/simlabrobotics/allegro_hand_ros_v4/blob/master/src/allegro_hand_description/allegro_hand_description_right.urdf
* Modified urdf to have 6 virtual joints, which allow us to move the gripper in 6 DOF (used for shaking controller in validation) 

# BELOW: Previous README

## Introduction

![Teaser](./images/teaser.png)

Robotic dexterous grasping is the first step to enable human-like dexterous object manipulation and thus a crucial robotic technology. However, dexterous grasping is much more under-explored than object grasping with parallel grippers, partially due to the lack of a large-scale dataset. In this work, we present a large-scale robotic dexterous grasp dataset, DexGraspNet, generated by our proposed highly efficient synthesis method that can be generally applied to any dexterous hand. Our method leverages a deeply accelerated differentiable force closure estimator and thus can efficiently and robustly synthesize stable and diverse grasps on a large scale. We choose ShadowHand and generate 1.32 million grasps for 5355 objects, covering more than 133 object categories and containing more than 200 diverse grasps for each object instance, with all grasps having been validated by the Isaac Gym simulator. Compared to the previous dataset from Liu et al. generated by GraspIt!, our dataset has not only more objects and grasps, but also higher diversity and quality. Via performing cross-dataset experiments, we show that training several algorithms of dexterous grasp synthesis on our dataset significantly outperforms training on the previous one.

## Qualitative Results

Some diverse grasps on the objects from DexGraspNet:

![QualitativeResults](./images/qualitative_results.png)

Our synthesis method can be applied to other robotic dexterous hands and the human hand. We provide the complete synthesis pipelines for Allegro and MANO in branches `allegro` and `mano` of this repo. Here are some results: 

![MultiHands](./images/multi_hands.png)

## Overview

This repository provides:

- Simple tools for visualizing grasp data.
- Asset processing for object models. See folder `asset_process`.
- Grasp generation. See folder `grasp_generation`.
  - We also updated code for
    - MANO grasp generation
    - Allegro grasp generation
    - ShadowHand grasp generation for objects on the table
  - See other branches for more information [TODO: update documents].

Our working file structure is as:

```bash
DexGraspNet
+-- asset_process
+-- grasp_generation
+-- data
|  +-- meshdata  # Linked to the output folder of asset processing.
|  +-- experiments  # Linked to a folder in the data disk. Small-scale experimental results go here.
|  +-- graspdata  # Linked to a folder in the data disk. Large-scale generated grasps go here, waiting for grasp validation.
|  +-- dataset  # Linked to a folder in the data disk. Validated results go here.
+-- thirdparty
|  +-- pytorch_kinematics
|  +-- CoACD
|  +-- ManifoldPlus
|  +-- TorchSDF
```

## Quick Example

```bash
conda create -n your_env python=3.7
conda activate your_env

# for quick example, cpu version is OK.
conda install pytorch cpuonly -c pytorch
conda install ipykernel
conda install transforms3d
conda install trimesh
pip install pyyaml
pip install lxml

cd thirdparty/pytorch_kinematics
pip install -e .
```

Then you can run `grasp_generation/quick_example.ipynb`.

For the full DexGraspNet dataset, go to our [project page](https://pku-epic.github.io/DexGraspNet/) for download links. Decompress dowloaded packages and link (or move) them to corresponding path in `data`.

## Citation

If you find our work useful in your research, please consider citing:

```
@article{wang2022dexgraspnet,
  title={DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation},
  author={Wang, Ruicheng and Zhang, Jialiang and Chen, Jiayi and Xu, Yinzhen and Li, Puhao and Liu, Tengyu and Wang, He},
  journal={arXiv preprint arXiv:2210.02697},
  year={2022}
}
```

## Contact

If you have any questions, please open a github issue or contact us:

Ruicheng Wang: <wrc0326@stu.pku.edu.cn>, Jialiang Zhang: <jackzhang0906@126.com>, He Wang: <hewang@pku.edu.cn>
