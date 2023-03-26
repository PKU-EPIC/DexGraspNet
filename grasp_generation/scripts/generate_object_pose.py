"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: Generate object pose, random free-fall, use SAPIEN
"""
import os

os.chdir(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import transforms3d
from multiprocessing import Pool
import sapien.core as sapien
from tqdm import tqdm


def generate_object_pose(_):

    args, object_code = _

    if not args.overwrite and os.path.exists(os.path.join(args.poses, object_code + '.npy')):
        return

    # set simulator

    engine = sapien.Engine()
    # engine.set_log_level('error')

    scene = engine.create_scene()
    scene.set_timestep(args.time_step)
    scene_config = sapien.SceneConfig()
    scene_config.default_restitution = args.restitution

    scene.add_ground(altitude=0, render=False)

    # load object

    if os.path.exists(os.path.join(args.data_root_path, object_code, 'coacd', 'coacd.urdf')) and not os.path.exists(os.path.join(args.data_root_path, object_code, 'coacd', 'coacd_convex_piece_63.obj')):
        loader = scene.create_urdf_loader()
        loader.fix_root_link = False
        object_actor = loader.load(os.path.join(args.data_root_path, object_code, 'coacd', 'coacd.urdf'))
        object_actor.set_name(name='object')
    else:
        builder = scene.create_actor_builder()
        builder.add_collision_from_file(os.path.join(args.data_root_path, object_code, 'coacd', 'decomposed.obj'))
        object_actor = builder.build(name='object')

    # generate object pose
    pose_matrices = []
    for i in range(args.n_samples):
        # random pose
        translation = np.zeros(3)
        translation[2] = 1 + np.random.rand()
        rotation_euler = 2 * np.pi * np.random.rand(3)
        rotation_quaternion = transforms3d.euler.euler2quat(*rotation_euler, axes='sxyz')
        try:
            object_actor.set_root_pose(sapien.Pose(translation, rotation_quaternion))
        except AttributeError:
            object_actor.set_pose(sapien.Pose(translation, rotation_quaternion))
        # simulate
        for t in range(args.sim_steps):
            scene.step()
        pose_matrices.append(object_actor.get_pose().to_transformation_matrix())
    pose_matrices = np.stack(pose_matrices)

    # save results

    np.save(os.path.join(args.poses, object_code + '.npy'), pose_matrices)
    
    # remove convex hull
    if os.path.exists(os.path.join(args.data_root_path, object_code, 'coacd', 'decomposed.obj.convex.stl')):
        os.remove(os.path.join(args.data_root_path, object_code, 'coacd', 'decomposed.obj.convex.stl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--data_root_path', type=str, default='../data/meshdata')
    parser.add_argument('--poses', type=str, default='../data/poses')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--overwrite', type=bool, default=True)
    parser.add_argument('--n_cpu', type=int, default=16)
    # simulator settings
    parser.add_argument('--sim_steps', type=int, default=1000)
    parser.add_argument('--time_step', type=float, default=1 / 100)
    parser.add_argument('--restitution', type=float, default=0.01)

    args = parser.parse_args()

    # seed
    np.random.seed(args.seed)

    # load object list
    object_code_list = os.listdir(args.data_root_path)

    # generate object pose
    # for object_code in tqdm(object_code_list, desc='generating'):
    #     generate_object_pose((args, object_code))
    with Pool(args.n_cpu) as p:
        param_list = []
        for object_code in object_code_list:
            param_list.append((args, object_code))
        list(tqdm(p.imap(generate_object_pose, param_list), desc='generating', total=len(param_list), miniters=1))
