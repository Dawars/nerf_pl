"""
Render validation images and calculate PSNR
Using sky mask and static image
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from einops import repeat
from tqdm import tqdm

from utils import load_ckpt
from collections import defaultdict
import matplotlib.pyplot as plt

from models.rendering import *
from models.nerf import *

import metrics

from datasets import dataset_dict

torch.backends.cudnn.benchmark = True

out_root = Path("/mnt/hdd/3d_recon/nerfw/eval")
ckpt_root = Path("/mnt/hdd/3d_recon/nerfw/")
dataset_path = Path("/mnt/hdd/3d_recon/sdfstudio/data/heritage/brandenburg_gate/")
mask_root = Path("/home/dawars/projects/master_thesis/NeuralRecon-W/output")

# Change to your settings...
############################
N_vocab = 1500
encode_appearance = True
N_a = 48
encode_transient = True
N_tau = 16
beta_min = 0.03  # doesn't have effect in testing

epochs = [0, 4, 9, 14, 19]
variations = list(range(4))
splits = list(range(7))


def apply_colormap(image, cmap="viridis"):
    """Convert single channel to a color image.

    Args:
        image: Single channel image.
        cmap: Colormap for image.

    Returns:
        TensorType: Colored image
    """

    colormap = plt.colormaps[cmap]
    image = np.nan_to_num(image, nan=0)
    image_long = (255 * np.array(image)).astype("int")
    image_long_min = np.min(image_long)
    image_long_max = np.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return np.array(colormap.colors)[image_long]


def apply_depth_colormap(
        depth,
        accumulation=None,
        near_plane: Optional[float] = None,
        far_plane: Optional[float] = None,
        cmap="turbo",
):
    """Converts a depth image to color for easier analysis.

    Args:
        depth: Depth image.
        accumulation: Ray accumulation used for masking vis.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.
        cmap: Colormap to apply.

    Returns:
        Colored depth image
    """

    near_plane = near_plane or float(np.min(depth))
    far_plane = far_plane or float(np.max(depth))

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = np.clip(depth, 0, 1)

    colored_image = apply_colormap(depth, cmap=cmap)

    if accumulation is not None:
        colored_image = colored_image * accumulation + (1 - accumulation)

    return colored_image


def eval_save(epoch, variation, split):
    """
    epoch: 0-19
    variation: random seed variation, 4 independent runs for same config
    split: reduce_images 0-6
    """

    exp_name = f"brandenburg_nerfw_2_{variation}_{split}"
    ckpt_path = ckpt_root / exp_name / f'epoch={epoch}.ckpt'
    if not ckpt_path.exists():
        print(f"CKPT DOESN'T EXIST {exp_name} {epoch}")
        return
    out_dir = out_root / exp_name / f"epoch{epoch}"
    out_dir.mkdir(exist_ok=True, parents=True)

    N_emb_xyz = 10
    N_emb_dir = 4
    N_samples = 128
    N_importance = 128
    use_disp = False
    chunk = 1024 * 32
    #############################

    embedding_xyz = PosEmbedding(N_emb_xyz - 1, N_emb_xyz)
    embedding_dir = PosEmbedding(N_emb_dir - 1, N_emb_dir)
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}
    if encode_appearance:
        embedding_a = torch.nn.Embedding(N_vocab, N_a).cuda()
        load_ckpt(embedding_a, ckpt_path, model_name='embedding_a')
        embeddings['a'] = embedding_a
    if encode_transient:
        embedding_t = torch.nn.Embedding(N_vocab, N_tau).cuda()
        load_ckpt(embedding_t, ckpt_path, model_name='embedding_t')
        embeddings['t'] = embedding_t

    nerf_coarse = NeRF('coarse',
                       in_channels_xyz=6 * N_emb_xyz + 3,
                       in_channels_dir=6 * N_emb_dir + 3).cuda()
    nerf_fine = NeRF('fine',
                     in_channels_xyz=6 * N_emb_xyz + 3,
                     in_channels_dir=6 * N_emb_dir + 3,
                     encode_appearance=encode_appearance,
                     in_channels_a=N_a,
                     encode_transient=encode_transient,
                     in_channels_t=N_tau,
                     beta_min=beta_min).cuda()

    load_ckpt(nerf_coarse, ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, ckpt_path, model_name='nerf_fine')

    models = {'coarse': nerf_coarse, 'fine': nerf_fine}

    @torch.no_grad()
    def f(rays, ts, **kwargs):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, chunk):
            kwargs_ = {}
            if 'a_embedded' in kwargs:
                kwargs_['a_embedded'] = kwargs['a_embedded'][i:i + chunk]
            rendered_ray_chunks = \
                render_rays(models,
                            embeddings,
                            rays[i:i + chunk],
                            ts[i:i + chunk],
                            N_samples,
                            use_disp,
                            0,
                            0,
                            N_importance,
                            chunk,
                            dataset.white_back,
                            test_time=True,
                            **kwargs_)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v.cpu()]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    dataset = dataset_dict['phototourism'](dataset_path,
                                           split='test_train',
                                           img_downscale=1, use_cache=False,
                                           reduce_images=0
                                           )

    avg_embedding = embedding_a(torch.arange(0, len(dataset), device="cuda")).mean(0)

    metrics_dict = {}

    for sample in tqdm(dataset):
        rays = sample['rays'].cuda()
        ts = sample['ts'].cuda()

        filename = dataset.image_paths[ts.cpu()[0].item()]
        sky_mask = np.array(Image.open(mask_root / f"{filename}_mask.png"))

        avg_embedding_ = repeat(avg_embedding, 'c -> (n1) c', n1=len(ts))
        results = f(rays, ts, a_embedded=avg_embedding_, output_transient=False)  # .unsqueeze(-1).tile(len(ts))

        img_wh = tuple(sample['img_wh'].numpy())
        img_gt = sample['rgbs'].view(img_wh[1], img_wh[0], 3)
        img_pred = results['rgb_fine'].view(img_wh[1], img_wh[0], 3).cpu().numpy()
        depth_pred = results['depth_fine'].view(img_wh[1], img_wh[0])

        psnr = metrics.psnr(img_gt, img_pred).item()
        psnr_mask = metrics.psnr(img_gt, img_pred, valid_mask=sky_mask).item()
        metrics_dict[filename] = {"mask": psnr_mask, "full": psnr}
        # print('PSNR between GT and pred:', psnr_mask, '\n')

        if encode_transient:
            # print('Decomposition--------------------------------------------' +
            #       '---------------------------------------------------------' +
            #       '---------------------------------------------------------' +
            #       '---------------------------------------------------------')
            beta = results['beta'].view(img_wh[1], img_wh[0]).cpu().numpy()
            img_pred_static = results['rgb_fine_static'].view(img_wh[1], img_wh[0], 3).cpu().numpy()
            img_pred_transient = results['_rgb_fine_transient'].view(img_wh[1], img_wh[0], 3).cpu().numpy()
            depth_pred_static = results['depth_fine_static'].view(img_wh[1], img_wh[0])
            depth_pred_transient = results['depth_fine_transient'].view(img_wh[1], img_wh[0])

            img_save_dir = (out_dir / filename).with_suffix("")
            plt.imsave(f"{img_save_dir}_rgb.png", img_pred_static)

            static_depth = depth_pred_static.cpu().numpy()
            near_plane = static_depth[sky_mask].min() - 0.1
            far_plane = static_depth[sky_mask].max() + 0.1
            plt.imsave(f"{img_save_dir}_depth.png", apply_depth_colormap(static_depth, near_plane=near_plane,
                                                                         far_plane=far_plane))

            # fig, (ax1, ax2) = plt.subplots(2)
            # ax1.imshow(img_gt)
            # ax2.imshow(img_pred_static)
            # plt.savefig(str(out_dir / filename))
            # plt.close()
        break

    (out_dir / "metrics.json").write_text(json.dumps(metrics_dict))
    print(f"Run done {epoch} {variation} {split}")


def main():
    for epoch in epochs:
        for variation in variations:
            for split in splits:
                eval_save(epoch, variation, split)


if __name__ == '__main__':
    main()
    # eval_save(19, 0, 0)
