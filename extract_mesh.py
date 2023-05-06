import torch
import os
import numpy as np
import cv2
import trimesh
from PIL import Image
from collections import defaultdict

from torch.nn import Embedding
from tqdm import tqdm
import mcubes
import open3d as o3d
from plyfile import PlyData, PlyElement
from argparse import ArgumentParser

from models.rendering import *
from models.nerf import *

from utils import load_ckpt

from datasets import dataset_dict

torch.backends.cudnn.benchmark = True


def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default="/mnt/hdd/3d_recon/sdfstudio/data/heritage/brandenburg_gate/",
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='phototourism',
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='gate_test',
                        help='scene name, used as output ply filename')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')

    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of samples to infer the acculmulated opacity')
    parser.add_argument('--chunk', type=int, default=32 * 1024,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--ckpt_path', type=str,
                        default="/mnt/hdd/3d_recon/nerfw/ckpts/nerfw_brandenburg_2_test512_1_0/epoch=19.ckpt",
                        help='pretrained checkpoint path to load')

    parser.add_argument('--N_grid', type=int, default=256,
                        help='size of the grid on 1 side, larger=higher resolution')
    parser.add_argument('--x_range', nargs="+", type=float, default=[-1.0, 1.0],
                        help='x range of the object')
    parser.add_argument('--y_range', nargs="+", type=float, default=[-1.0, 1.0],
                        help='x range of the object')
    parser.add_argument('--z_range', nargs="+", type=float, default=[-1.0, 1.0],
                        help='x range of the object')
    parser.add_argument('--sigma_threshold', type=float, default=10.0,
                        help='threshold to consider a location is occupied')
    parser.add_argument('--occ_threshold', type=float, default=0.2,
                        help='''threshold to consider a vertex is occluded.
                                larger=fewer occluded pixels''')

    #### method using vertex normals ####
    parser.add_argument('--use_vertex_normal', action="store_true",
                        help='use vertex normals to compute color')
    parser.add_argument('--N_importance', type=int, default=64,
                        help='number of fine samples to infer the acculmulated opacity')
    parser.add_argument('--near_t', type=float, default=1.0,
                        help='the near bound factor to start the ray')

    return parser.parse_args()


radius = 4.6
origin = np.array([0.568699, -0.0935532, 6.28958])
scale = 9.360126495361328
bb = np.array([origin - radius,
               origin + radius]) / scale


@torch.no_grad()
def f(models, embeddings, rays, N_samples, N_importance, chunk, white_back):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i + chunk],
                        N_samples,
                        False,
                        0,
                        0,
                        N_importance,
                        chunk,
                        white_back,
                        test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    args = get_opts()

    kwargs = {'root_dir': args.root_dir,
              # 'img_wh': tuple(args.img_wh)
              }
    if args.dataset_name == 'llff':
        kwargs['spheric_poses'] = True
        kwargs['split'] = 'test'
    else:
        kwargs['split'] = 'train'
        kwargs['img_downscale'] = 2
        kwargs['use_cache'] = False
    dataset = dataset_dict[args.dataset_name](**kwargs)

    N_vocab = 1500
    encode_appearance = True
    N_a = 48
    encode_transient = True
    N_tau = 16
    beta_min = 0.03  # doesn't have effect in testing

    N_emb_xyz = 10
    N_emb_dir = 4
    N_samples = 128
    N_importance = 128
    use_disp = False

    embedding_xyz = PosEmbedding(N_emb_xyz - 1, N_emb_xyz)
    embedding_dir = PosEmbedding(N_emb_dir - 1, N_emb_dir)
    embedding_xyz.cuda()
    embedding_dir.cuda()
    embeddings = [embedding_xyz, embedding_dir]
    nerf_fine = NeRF("fine", W=512,
                     in_channels_xyz=6 * N_emb_xyz + 3,
                     in_channels_dir=6 * N_emb_dir + 3,
                     encode_appearance=encode_appearance,
                     in_channels_a=N_a,
                     encode_transient=encode_transient,
                     in_channels_t=N_tau,
                     beta_min=beta_min
                     )
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
    nerf_fine.cuda().eval()

    # define the dense grid for query
    N = args.N_grid
    xmin, xmax = bb[0, 0], bb[1, 0]
    ymin, ymax = bb[0, 1], bb[1, 1]
    zmin, zmax = bb[0, 2], bb[1, 2]
    # assert xmax-xmin == ymax-ymin == zmax-zmin, 'the ranges must have the same length!'
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)

    xyz_ = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)).cuda()
    dir_ = torch.zeros_like(xyz_).cuda()
    # sigma is independent of direction, so any value here will produce the same result

    # predict sigma (occupancy) for each grid location
    print('Predicting occupancy ...')
    with torch.no_grad():
        B = xyz_.shape[0]
        out_chunks = []
        for i in tqdm(range(0, B, args.chunk)):
            xyz_embedded = embedding_xyz(xyz_[i:i + args.chunk])  # (N, embed_xyz_channels)
            dir_embedded = embedding_dir(dir_[i:i + args.chunk])  # (N, embed_dir_channels)
            a_embedded = torch.zeros(xyz_embedded.shape[0], N_a, device=xyz_embedded.device)
            t_embedded = torch.zeros(xyz_embedded.shape[0], N_tau, device=xyz_embedded.device)
            xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded, a_embedded, t_embedded], 1)
            out_chunks += [nerf_fine(xyzdir_embedded).cpu()]
        rgbsigma = torch.cat(out_chunks, 0)

    sigma = rgbsigma[:, -1].cpu().numpy()
    sigma = np.maximum(sigma, 0).reshape(N, N, N)

    # perform marching cube algorithm to retrieve vertices and triangle mesh
    print('Extracting mesh ...')
    vertices, triangles = mcubes.marching_cubes(sigma, args.sigma_threshold)

    ##### Until mesh extraction here, it is the same as the original repo. ######

    vertices_ = ((vertices) / N - (origin[[1, 0, 2]] / scale)).astype(np.float32)
    ## invert x and y coordinates (WHY? maybe because of the marching cubes algo)
    x_ = (ymax - ymin) * vertices_[:, 1] + ymin
    y_ = (xmax - xmin) * vertices_[:, 0] + xmin
    vertices_[:, 0] = x_ * scale
    vertices_[:, 1] = y_ * scale
    vertices_[:, 2] = ((zmax - zmin) * vertices_[:, 2] + zmin) * scale
    vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

    face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = triangles
    PlyData([PlyElement.describe(vertices_[:, 0], 'vertex'),
             PlyElement.describe(face, 'face')]).write(f'{args.scene_name}.ply')

    # remove noise in the mesh by keeping only the biggest cluster
    # print('Removing noise ...')
    # mesh = o3d.io.read_triangle_mesh(f"{args.scene_name}.ply")
    # idxs, count, _ = mesh.cluster_connected_triangles()
    # max_cluster_idx = np.argmax(count)
    # triangles_to_remove = [i for i in range(len(face)) if idxs[i] != max_cluster_idx]
    # mesh.remove_triangles_by_index(triangles_to_remove)
    # mesh.remove_unreferenced_vertices()
    # print(f'Mesh has {len(mesh.vertices) / 1e6:.2f} M vertices and {len(mesh.triangles) / 1e6:.2f} M faces.')
    #
    # o3d.io.write_triangle_mesh(f"{args.scene_name}_smooth.ply", mesh)
