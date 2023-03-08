# Asset process

This folder is for processing object models. From different object datasets, we choose suitable categories, filer out non-manifolds and models of small volume, and decompose them into convex pieces for physical simulation.

## Dependencies

### ManifoldPlus

Following README in [ManifoldPlus](https://github.com/hjwdzh/ManifoldPlus):

```bash
git clone https://github.com/hjwdzh/ManifoldPlus.git
cd ManifoldPlus
git submodule update --init --recursive
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

### CoACD

Instead of VHACD, we use [CoACD](https://github.com/SarahWeiii/CoACD) as our tool of approximate convex decompostion.

```bash
git clone --recurse-submodules https://github.com/SarahWeiii/CoACD.git
cd CoACD
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

## Usage

We process our object models as the following pipeline. If you have some object models and want to synthesise grasps, we recommand you to follow the pipeline to process these models too.

- Extraction
  - Organize models into a folder.
- Manifold
  - Use ManifoldPlus to convert raw models into manifolds robustly.
- Normalization
  - Adjust centers and sizes of models. Then filter out bad models.
- Decomposition
  - Use CoACD to decompose models and export urdf files for later physical simulation.

Below are sources of our object datasets:

- [ShapeNetCore](https://shapenet.org/)
- [ShapeNetSem](https://shapenet.org/)
- [Mujoco](https://github.com/kevinzakka/mujoco_scanned_objects)
- [DDG](https://gamma.umd.edu/researchdirections/grasping/differentiable_grasp_planner)(Deep Differentiable Grasp)

### Extraction

```bash
# ShapeNetCore
python extract.py --src data/ShapeNetCore.v2 --dst data/raw_models --set core
# ShapeNetSem
python extract.py --src data/ShapeNetSem/models --dst data/raw_models --set sem --meta data/ShapeNetSem/metadata.csv
# Mujoco
python extract.py --src data/mujoco_scanned_objects/models --dst data/raw_models --set mujoco
# DDG
python extract.py --src data/Grasp_Dataset/good_shapes --dst data/raw_models --set ddg
```

### Manifold

```bash
python manifold.py --src data/raw_models --dst data/manifolds --manifold_path ./ManifoldPlus/build/manifold
```

This generates `run.sh`. Then run it with:

```bash
bash run.sh
# or poolrun.py runs it in multiprocess
python poolrun.py -p 32
```

### Normalization

```bash
python normalize.py --src data/manifolds --dst data/normalized_models
```

### Decomposition

```bash
python decompose_list.py --src data/normalized_models --dst data/meshdata --coacd_path ./CoACD/build/main
```

Again this generates `run.sh`.

```bash
bash run.sh
# or
python poolrun.py -p 32
```

The structure of the final dataset is:

- meshdata
  - source(-category)-code0
    - coacd
      - coacd_convex_piece_0.obj
      - coacd_convex_piece_1.obj
      - ...
      - coacd.urdf
      - decomposed.obj
      - ...
  - source(-category)-code1
  - ...
