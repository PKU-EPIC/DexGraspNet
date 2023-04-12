import argparse
import trimesh
import numpy as np
import tqdm
from utils.extract_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    for code in tqdm(os.listdir(args.src)):
        mesh = trimesh.load(os.path.join(args.src, code),
                            force="mesh", process=False)
        verts = np.array(mesh.vertices)
        xcenter = (np.max(verts[:, 0]) + np.min(verts[:, 0])) / 2
        ycenter = (np.max(verts[:, 1]) + np.min(verts[:, 1])) / 2
        zcenter = (np.max(verts[:, 2]) + np.min(verts[:, 2])) / 2
        verts_ = verts - np.array([xcenter, ycenter, zcenter])
        dmax = np.max(np.sqrt(np.sum(np.square(verts_), axis=1))) * 1.03
        verts_ /= dmax
        mesh_ = trimesh.Trimesh(
            vertices=verts_, faces=mesh.faces, process=False)
        if(mesh_.is_watertight and mesh_.volume > 0.05):
            mesh_.export(os.path.join(args.dst, code))
