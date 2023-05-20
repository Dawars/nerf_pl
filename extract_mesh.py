from math import ceil
from pathlib import Path

import pymeshlab
import numpy as np
import trimesh
from collections import defaultdict

import yaml
from skimage import measure
from tqdm import tqdm

from models.rendering import *
from models.nerf import *

from utils import load_ckpt

torch.backends.cudnn.benchmark = True

torch.set_float32_matmul_precision("high")

simplify = False
# coarse_mask = torch.tensor(np.load("/home/dawars/projects/sdfstudio/mask.npy"))
coarse_mask = None
if coarse_mask is not None:
    # we need to permute here as pytorch's grid_sample use (z, y, x)
    coarse_mask = coarse_mask.permute(2, 1, 0)[None, None].cuda().float()
#
# radius = 4.6
# origin = np.array([0.568699, -0.0935532, 6.28958])
# scale = 11.384292602539062
# bb_neus = np.array([origin - radius, origin + radius]) / scale
# extends = np.array([4.6, 2.16173, 4.6])
# bb = np.array([origin - extends, origin + extends]) / scale
# with open("/mnt/hdd/3d_recon/neural_recon_w/jena/observatorium/config.yaml", "r") as yamlfile:
with open("/mnt/hdd/3d_recon/neural_recon_w/heritage-recon/brandenburg_gate/config.yaml", "r") as yamlfile:
    scene_config = yaml.load(yamlfile, Loader=yaml.FullLoader)

radius = scene_config["radius"]

origin = np.array(scene_config["origin"])
scale = 11.384292602539062
bb_neus = np.array([origin - scale, origin + scale]) / scale
extends = np.array([4.6, 2.16173, 4.6])
bb = np.array([origin - extends, origin + extends]) / scale
# brandenburg
sfm_to_gt = np.array(scene_config["sfm2gt"])
gt_to_sfm = np.linalg.inv(sfm_to_gt)
sfm_vert1 = gt_to_sfm[:3, :3] @ np.array(scene_config["eval_bbx"][0]) + gt_to_sfm[:3, 3]
sfm_vert2 = gt_to_sfm[:3, :3] @ np.array(scene_config["eval_bbx"][1]) + gt_to_sfm[:3, 3]
bbx_min = np.minimum(sfm_vert1, sfm_vert2)
bbx_max = np.maximum(sfm_vert1, sfm_vert2)

bb_conf = np.stack([bbx_min, bbx_max], axis=1).T / scale
bb = bb_conf

# avg_pool_3d = torch.nn.AvgPool3d(2, stride=2)
upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")

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
nerf_fine = NeRF("fine", W=256,
                 in_channels_xyz=6 * N_emb_xyz + 3,
                 in_channels_dir=6 * N_emb_dir + 3,
                 encode_appearance=encode_appearance,
                 in_channels_a=N_a,
                 encode_transient=encode_transient,
                 in_channels_t=N_tau,
                 beta_min=beta_min
                 )

# define the dense grid for query
N = 512
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


@torch.no_grad()
def evaluate(points):
    z_ = []
    for _, pnts in enumerate(tqdm(torch.split(points, 100000, dim=0))):
        xyz_embedded = embedding_xyz(pnts)  # (N, embed_xyz_channels)
        dir_embedded = embedding_dir(torch.zeros_like(pnts).cuda())  # (N, embed_dir_channels)
        a_embedded = torch.zeros(xyz_embedded.shape[0], N_a, device=xyz_embedded.device)
        t_embedded = torch.zeros(xyz_embedded.shape[0], N_tau, device=xyz_embedded.device)
        xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded, a_embedded, t_embedded], 1)
        z_.append(nerf_fine(xyzdir_embedded).cpu()[:, -1])
    z_ = torch.cat(z_, axis=0).cuda()
    return z_


def extract_mesh(epoch: int, split: int):
    # global mask
    global coarse_mask
    # checkpoint_path = Path(f"/mnt/hdd/3d_recon/nerfw/ckpts/nerfw_brandenburg_2_{variation}_{split}/epoch={epoch}.ckpt")
    checkpoint_path = Path(f"/mnt/hdd/3d_recon/nerfw/ckpts/nerfw_gate_{split}/epoch={epoch}.ckpt")
    load_ckpt(nerf_fine, str(checkpoint_path), model_name='nerf_fine')
    nerf_fine.cuda().eval()

    # predict sigma (occupancy) for each grid location
    print('Predicting occupancy ...')

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()

    # construct point pyramids
    points = points.reshape(N, N, N, 3).permute(3, 0, 1, 2)
    if coarse_mask is not None:
        # breakpoint()
        points_tmp = points.permute(1, 2, 3, 0)[None].cuda()
        current_mask = torch.nn.functional.grid_sample(coarse_mask, points_tmp)
        current_mask = (current_mask > 0.0).cpu().numpy()[0, 0]
    else:
        current_mask = None

    points_pyramid = [points]
    # for _ in range(3):
    #     points = avg_pool_3d(points[None])[0]
    #     points_pyramid.append(points)
    # points_pyramid = points_pyramid[::-1]

    # evalute pyramid with mask
    mask = None
    threshold = 2 * (xmax - xmin) / N * 8
    for pid, pts in enumerate(points_pyramid):
        coarse_N = pts.shape[-1]
        pts = pts.reshape(3, -1).permute(1, 0).contiguous()

        if mask is None:
            # only evaluate
            if coarse_mask is not None:
                pts_sdf = torch.ones_like(pts[:, 1])
                valid_mask = (
                        torch.nn.functional.grid_sample(coarse_mask, pts[None, None, None])[0, 0, 0, 0] > 0
                )
                if valid_mask.any():
                    pts_sdf[valid_mask] = evaluate(pts[valid_mask].contiguous())
            else:
                pts_sdf = evaluate(pts)
        else:
            mask = mask.reshape(-1)
            pts_to_eval = pts[mask]

            if pts_to_eval.shape[0] > 0:
                pts_sdf_eval = evaluate(pts_to_eval.contiguous())
                pts_sdf[mask] = pts_sdf_eval
            print("ratio", pts_to_eval.shape[0] / pts.shape[0])

        # if pid < 3:
        # #     update mask
        # mask = torch.abs(pts_sdf) < threshold
        # mask = mask.reshape(coarse_N, coarse_N, coarse_N)[None, None]
        # mask = upsample(mask.float()).bool()
        #
        # pts_sdf = pts_sdf.reshape(coarse_N, coarse_N, coarse_N)[None, None]
        # pts_sdf = upsample(pts_sdf)
        # pts_sdf = pts_sdf.reshape(-1)

        threshold /= 2.0

    pts_sdf = pts_sdf.detach().cpu().numpy()

    # skip if no surface found
    # if current_mask is not None:
    #     valid_z = z.reshape(N, N, N)[current_mask]
    #     if valid_z.shape[0] <= 0 or (np.min(valid_z) > level or np.max(valid_z) < level):
    #         break  # todo loop
    print(f"{np.min(pts_sdf)} {np.max(pts_sdf)}")

    for level in tqdm(range(1, ceil(pts_sdf.max()))):
        # for level in tqdm([10]):
        if not (np.min(pts_sdf) > level or np.max(pts_sdf) < level):
            out_path = checkpoint_path.with_name(f"mesh_{epoch}_{level}.ply")
            out_path_simplify = checkpoint_path.with_name(f"mesh_{epoch}_{level}_simple.ply")

            pts_sdf = pts_sdf.astype(np.float32)
            verts, faces, normals, _ = measure.marching_cubes(
                volume=pts_sdf.reshape(N, N, N),  # .transpose([1, 0, 2]),
                level=level,
                spacing=(
                    (xmax - xmin) / (N - 1),
                    (ymax - ymin) / (N - 1),
                    (zmax - zmin) / (N - 1),
                ),
                mask=current_mask,
            )
            verts = verts + np.array([xmin, ymin, zmin])

            meshcrop = trimesh.Trimesh(verts * scale, faces, normals)
            meshcrop.export(str(out_path))

            # simplify
            if simplify:
                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(str(out_path))

                ms.meshing_decimation_quadric_edge_collapse(targetfacenum=2_000_000)
                ms.save_current_mesh(str(out_path_simplify), save_face_color=False)
        else:
            print(f"Level not in min max {split}_{epoch}_{level}")


if __name__ == "__main__":
    # extract_mesh(19, 0)
    epoch = 19
    for split in range(7):
        extract_mesh(19, split)
        print(f"{split}: {epoch}")
        extract_mesh(epoch, split)
