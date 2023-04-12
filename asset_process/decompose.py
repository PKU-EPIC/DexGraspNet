import os
import shutil
from distutils.spawn import find_executable
import numpy as np
from trimesh.version import __version__ as trimesh_version
from trimesh.exchange.obj import load_obj
import trimesh as tm
import argparse


def export_urdf(
        coacd_path, input_filename,
        output_directory,
        scale=1.0,
        color=[0.75, 0.75, 0.75],
        **kwargs):
    """
    Convert a Trimesh object into a URDF package for physics simulation.
    This breaks the mesh into convex pieces and writes them to the same
    directory as the .urdf file.

    Parameters
    ---------
    input_filename   : str
    output_directiry : str
                  The directory path for the URDF package

    Returns
    ---------
    mesh : Trimesh object
             Multi-body mesh containing convex decomposition
    """

    import lxml.etree as et
    # TODO: fix circular import
    from trimesh.exchange.export import export_mesh
    # Extract the save directory and the file name
    fullpath = os.path.abspath(output_directory)
    name = os.path.basename(fullpath)
    _, ext = os.path.splitext(name)

    if ext != '':
        raise ValueError('URDF path must be a directory!')

    # Create directory if needed
    if not os.path.exists(fullpath):
        os.mkdir(fullpath)
    elif not os.path.isdir(fullpath):
        raise ValueError('URDF path must be a directory!')

    # Perform a convex decomposition
    # if not exists:
    #     raise ValueError('No coacd available!')

    argstring = f' -i {input_filename} -o {os.path.join(output_directory, "decomposed.obj")}'

    # pass through extra arguments from the input dictionary
    for key, value in kwargs.items():
        argstring += ' -{} {}'.format(str(key),
                                      str(value))
    os.system(coacd_path + argstring)

    convex_pieces = list(tm.load(os.path.join(
        output_directory, 'decomposed.obj'), process=False).split())

    # Get the effective density of the mesh
    mesh = tm.load(input_filename,force="mesh", process=False)
    effective_density = mesh.volume / sum([
        m.volume for m in convex_pieces])

    # open an XML tree
    root = et.Element('robot', name='root')

    # Loop through all pieces, adding each as a link
    prev_link_name = None
    for i, piece in enumerate(convex_pieces):

        # Save each nearly convex mesh out to a file
        piece_name = '{}_convex_piece_{}'.format(name, i)
        piece_filename = '{}.obj'.format(piece_name)
        piece_filepath = os.path.join(fullpath, piece_filename)
        export_mesh(piece, piece_filepath)

        # Set the mass properties of the piece
        piece.center_mass = mesh.center_mass
        piece.density = effective_density * mesh.density

        link_name = 'link_{}'.format(piece_name)
        geom_name = '{}'.format(piece_filename)
        I = [['{:.2E}'.format(y) for y in x]  # NOQA
             for x in piece.moment_inertia]

        # Write the link out to the XML Tree
        link = et.SubElement(root, 'link', name=link_name)

        # Inertial information
        inertial = et.SubElement(link, 'inertial')
        et.SubElement(inertial, 'origin', xyz="0 0 0", rpy="0 0 0")
        # et.SubElement(inertial, 'mass', value='{:.2E}'.format(piece.mass))
        et.SubElement(
            inertial,
            'inertia',
            ixx=I[0][0],
            ixy=I[0][1],
            ixz=I[0][2],
            iyy=I[1][1],
            iyz=I[1][2],
            izz=I[2][2])
        # Visual Information
        visual = et.SubElement(link, 'visual')
        et.SubElement(visual, 'origin', xyz="0 0 0", rpy="0 0 0")
        geometry = et.SubElement(visual, 'geometry')
        et.SubElement(geometry, 'mesh', filename=geom_name,
                      scale="{:.4E} {:.4E} {:.4E}".format(scale,
                                                          scale,
                                                          scale))
        material = et.SubElement(visual, 'material', name='')
        et.SubElement(material,
                      'color',
                      rgba="{:.2E} {:.2E} {:.2E} 1".format(color[0],
                                                           color[1],
                                                           color[2]))

        # Collision Information
        collision = et.SubElement(link, 'collision')
        et.SubElement(collision, 'origin', xyz="0 0 0", rpy="0 0 0")
        geometry = et.SubElement(collision, 'geometry')
        et.SubElement(geometry, 'mesh', filename=geom_name,
                      scale="{:.4E} {:.4E} {:.4E}".format(scale,
                                                          scale,
                                                          scale))

        # Create rigid joint to previous link
        if prev_link_name is not None:
            joint_name = '{}_joint'.format(link_name)
            joint = et.SubElement(root,
                                  'joint',
                                  name=joint_name,
                                  type='fixed')
            et.SubElement(joint, 'origin', xyz="0 0 0", rpy="0 0 0")
            et.SubElement(joint, 'parent', link=prev_link_name)
            et.SubElement(joint, 'child', link=link_name)

        prev_link_name = link_name

    # Write URDF file
    tree = et.ElementTree(root)
    urdf_filename = '{}.urdf'.format(name)
    tree.write(os.path.join(fullpath, urdf_filename),
               pretty_print=True)

    # Write Gazebo config file
    root = et.Element('model')
    model = et.SubElement(root, 'name')
    model.text = name
    version = et.SubElement(root, 'version')
    version.text = '1.0'
    sdf = et.SubElement(root, 'sdf', version='1.4')
    sdf.text = '{}.urdf'.format(name)

    author = et.SubElement(root, 'author')
    et.SubElement(author, 'name').text = 'trimesh {}'.format(trimesh_version)
    et.SubElement(author, 'email').text = 'blank@blank.blank'

    description = et.SubElement(root, 'description')
    description.text = name

    tree = et.ElementTree(root)
    tree.write(os.path.join(fullpath, 'model.config'))

    return np.sum(convex_pieces)


def decompose(args, object_code):

    print(f'decomposition: {object_code}')

    if os.path.exists(os.path.join(args.result_path, object_code, 'coacd')):
        shutil.rmtree(os.path.join(args.result_path, object_code, 'coacd'))
    os.makedirs(os.path.join(args.result_path, object_code, 'coacd'))
    coacd_params = {
        't': args.t,
        'k': args.k
    }
    export_urdf(args.coacd_path, os.path.join(args.data_root_path, object_code + ".obj"),
                os.path.join(args.result_path, object_code, 'coacd'), **coacd_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--k', default=0.3, type=float)
    parser.add_argument('--t', default=0.08, type=float)

    parser.add_argument('--coacd_path', required=True, type=str)
    parser.add_argument('--result_path', required=True, type=str)
    parser.add_argument('--data_root_path', required=True, type=str)
    parser.add_argument('--object_code', required=True, type=str)

    args = parser.parse_args()

    # check whether arguments are valid and process arguments

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    if not os.path.exists(args.data_root_path):
        raise ValueError(
            f'data_root_path {args.data_root_path} doesn\'t exist')

    decompose(args, args.object_code)
