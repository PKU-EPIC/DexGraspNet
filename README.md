# DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation

This is the official repository of [DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation](https://arxiv.org/abs/2210.02697).

[[project page]](https://pku-epic.github.io/DexGraspNet/)

## How to Run (Updated 2023-12-04)

### Run Full Pipeline (in development)

After following instructions below to set up the environment and the mesh dataset, create a new folder with the meshes you would like to generate grasps for (if run on default `../data/meshdata`, it will take very long running on all meshes). For example:

```
# Create new folder that symlinks to data in main meshdata folder (can also directly copy as well)
mkdir ../data/2023-12-04_meshdata_rubikscube_one_object
ln -rs ../data/meshdata/ddg-gd_rubik_cube_poisson_004 ../data/2023-12-04_meshdata_rubikscube_one_object/ddg-gd_rubik_cube_poisson_004
```

Then run the following:
```
time CUDA_VISIBLE_DEVICES=0 python scripts/generate_all_grasps.py --input_meshdata_path ../data/2023-12-04_meshdata_rubikscube_one_object/ddg-gd_rubik_cube_poisson_004 --experiment_name 2023-12-04_rubikscube_one_object --genera --generate_nerf_data --num_random_pose_noise_samples_per_grasp 5
```

This runs the full pipeline for all meshes in `../data/2023-12-04_meshdata_rubikscube_one_object` and generates nerfdata and tests each grasp in isaacgym validation 5 times (each with slight pose noise)

After running this, you will have the following rough directory structure in `2023-12-04_rubikscube_one_object`:

```
ls ../data/2023-12-04_rubikscube_one_object
augmented_raw_evaled_grasp_config_dicts_opened_hand  augmented_raw_hand_config_dicts_opened_hand  nerfdata
augmented_raw_grasp_config_dicts_opened_hand         evaled_grasp_config_dicts                    raw_evaled_grasp_config_dicts
augmented_raw_hand_config_dicts_closed_hand          hand_config_dicts                            raw_grasp_config_dicts
```

You are ready for nerf_grasping!

### Debugging Generate Hand Config Dicts

Add in wandb logging and modify `thres_dis`
```
CUDA_VISIBLE_DEVICES=0 python scripts/generate_hand_config_dicts.py --meshdata_root_path ../data/2023-12-04_meshdata_rubikscube_one_object/  --output_hand_config_dicts_path ../data/2023-12-04_debug/hand_config_dicts  --use_penetration_energy --thres_dis 0.2 --use_wandb         
```

Can view plots of energy vs. iteration and visualize the grasps on wandb.

### Debugging Isaac Validator

Change the `DEBUG` flag in `eval_grasp_config_dict.py` and/or `isaac_validator.py` to get more detailed debug info. 

You can visualize specific simulations like so (this looks at a specific augmented grasp from opening a mid optimization grasp):

```
CUDA_VISIBLE_DEVICES=0 python scripts/eval_grasp_config_dict.py --hand_model_type ALLEGRO_HAND --validation_type NO_GRAVITY_SHAKING --gpu 0 --meshdata_root_path ../data/meshdata --input_grasp_config_dicts_path ../data/2023-12-04_rubikscube_one_object/evaled_grasp_config_dicts --output_evaled_grasp_config_dicts_path ../data/2023-12-04_rubikscube_one_object/augmented_raw_evaled_grasp_config_dicts_opened_hand/mid_optimization/1800 --object_code_and_scale_str ddg-gd_rubik_cube_poisson_004_0_1000 --max_grasps_per_batch 5000 --debug_index 0 --use_gui
```

Move fingers back first (different strategy)
```
CUDA_VISIBLE_DEVICES=0 python scripts/eval_grasp_config_dict.py --hand_model_type ALLEGRO_HAND --validation_type NO_GRAVITY_SHAKING --gpu 0 --meshdata_root_path ../data/meshdata --input_grasp_config_dicts_path ../data/2023-12-04_rubikscube_one_object/evaled_grasp_config_dicts --output_evaled_grasp_config_dicts_path ../data/2023-12-04_rubikscube_one_object/augmented_raw_evaled_grasp_config_dicts_opened_hand/mid_optimization/1800 --object_code_and_scale_str ddg-gd_rubik_cube_poisson_004_0_1000 --max_grasps_per_batch 5000 --debug_index 0 --use_gui --move_fingers_back_at_init
```

Run for all grasps (can change `set_seed` to get different results to see reproducibility):
```
CUDA_VISIBLE_DEVICES=0 python scripts/eval_grasp_config_dict.py --hand_model_type ALLEGRO_HAND --validation_type NO_GRAVITY_SHAKING --gpu 0 --meshdata_root_path ../data/meshdata --input_grasp_config_dicts_path ../data/2023-12-04_rubikscube_one_object/evaled_grasp_config_dicts --output_evaled_grasp_config_dicts_path ../data/2023-12-04_rubikscube_one_object/augmented_raw_evaled_grasp_config_dicts_opened_hand/mid_optimization/1800 --object_code_and_scale_str ddg-gd_rubik_cube_poisson_004_0_1000 --max_grasps_per_batch 5000 --move_fingers_back_at_init
```

Some params to investigate:
```
NUM_STEPS_TO_NOT_MOVE_HAND_JOINTS = 10
NUM_STEPS_TO_CLOSE_HAND_JOINTS = 15
NUM_STEPS_TO_NOT_MOVE_WRIST_POSE = 30

DIST_MOVE_FINGER_BACKWARDS = -0.06
DIST_MOVE_FINGER = 0.1
```

### Visualize Tools

Visualize one grasp on one object:
```
python visualize/visualize_config_dict.py --input_config_dicts_path ../data/2023-12-04_rubikscube_one_object/evaled_grasp_config_dicts --meshdata_root_path ../data/meshdata --object_code_and_scale_str ddg-gd_rubik_cube_poisson_004_0_1000 --idx_to_visualize 0
```

Visualize multiple grasps on one object:
```
python visualize/visualize_config_dict.py --input_config_dicts_path ../data/2023-12-04_rubikscube_one_object/evaled_grasp_config_dicts --meshdata_root_path ../data/meshdata --object_code_and_scale_str ddg-gd_rubik_cube_poisson_004_0_1000 --visualize_all
```

Visualize the optimization of one grasp on one object (may only work if have a fixed frequency of mid_optimizations stored, set with `--store_grasps_mid_optimization_freq 25` for generate_hand_config_dicts.py):
```
python visualize/visualize_config_dict_optimization.py --input_config_dicts_mid_optimization_path ../data/2023-12-04_rubikscube_one_object/hand_config_dicts/mid_optimization --object_code_and_scale_str ddg-gd_rubik_cube_poisson_004_0_1000 --meshdata_root_path ../data/meshdata --idx_to_visualize 0
```

Visualize multiple meshes files (useful when looking at meshes generated by nerf):
```
python visualize_objs.py --meshdata_root_path ../data/meshdata
```

### Eval DGN Baseline

Compare how well this pipeline works with a ground truth mesh vs nerf-generated mesh. Assume you have folder `../data/2023-11-17_meshdata_mugs` with some mug meshes and `../../nerf_grasping/data/2023-11-17_01-27-23/nerf_meshdata_mugs` with some mug meshes generated from nerf (done in nerf_grasping since needs nerfstudio). Note: it assumes that nerf meshes are of the same scale as the ground truth meshes (done by default). They should have the same file structure like so:

```
tree ../data/2023-11-17_meshdata_mugs/core-mug-1038e4eac0e18dcce02ae6d2a21d494a 
../data/2023-11-17_meshdata_mugs/core-mug-1038e4eac0e18dcce02ae6d2a21d494a
└── coacd
    ├── coacd_convex_piece_0.obj
    ├── coacd_convex_piece_1.obj
    ├── coacd_convex_piece_2.obj
    ├── coacd_convex_piece_3.obj
    ├── coacd_convex_piece_4.obj
    ├── coacd_convex_piece_5.obj
    ├── coacd_convex_piece_6.obj
    ├── coacd_convex_piece_7.obj
    ├── coacd_convex_piece_8.obj
    ├── coacd_convex_piece_9.obj
    ├── coacd.urdf
    ├── decomposed_log.txt
    ├── decomposed.obj
    ├── decomposed.wrl
    └── model.config
```

Ground truth mesh:
```
time CUDA_VISIBLE_DEVICES=0 python eval_dgn_baseline.py --meshdata_root_path ../data/2023-11-17_meshdata_mugs --nerf_meshdata_root_path --output_eval_results_path ../data/eval_results/meshplan
```

Nerf generated mesh:
```
time CUDA_VISIBLE_DEVICES=0 python eval_dgn_baseline.py --meshdata_root_path ../data/2023-11-17_meshdata_mugs --nerf_meshdata_root_path ../../nerf_grasping/data/2023-11-17_01-27-23/nerf_meshdata_mugs --plan_using_nerf --output_eval_results_path ../data/eval_results/nerfplan
```

It will output something like:

```
Total success rate: num_successes / num_total = 63 / 2500 = 0.0252
Total success rate of top 10 energy: num_successes_best_k / num_total_best_k = 0 / 100 = 0.0
```

This shows success rate of all and success rate of the top N grasps with the lowest energy.

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

### Grasp Config Dict
Has the same as the hand_config_dict, but also has:

```
grasp_config_dict['grasp_orientations'].shape == (batch_size, n_fingers, 3, 3)
```

Note that `grasp_orientations` refer to rotation matrices that specify the direction and orientation that each finger should move along to complete a grasp, with the z-dim along the grasp approach direction and the y-dim along the finger to fingertip direction (modified to be perpendicular to z).

### Evaled Grasp Config Dict
Has the same as the grasp_config_dict, but also has:

```
evaled_grasp_config_dict['passed_eval'].shape == (batch_size,)
evaled_grasp_config_dict['passed_simulation'].shape == (batch_size,)
evaled_grasp_config_dict['passed_penetration_threshold'].shape == (batch_size,)
evaled_grasp_config_dict['penetration'].shape == (batch_size,)
```

### Optimized Grasp Config Dict
Has the same as the grasp_config_dict, but also has:

```
optimized_grasp_config_dict['scores'].shape == (batch_size,)
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
