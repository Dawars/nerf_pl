import os
from pathlib import Path

from tqdm import tqdm

from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict, PhototourismDataset

# models
from models.nerf import *
from models.rendering import *

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.loss = loss_dict['nerfw'](coef=1)

        self.models_to_train = []
        self.embedding_xyz = PosEmbedding(hparams.N_emb_xyz-1, hparams.N_emb_xyz)
        self.embedding_dir = PosEmbedding(hparams.N_emb_dir-1, hparams.N_emb_dir)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        if hparams.encode_a:
            self.embedding_a = torch.nn.Embedding(hparams.N_vocab, hparams.N_a)
            self.embeddings['a'] = self.embedding_a
            self.models_to_train += [self.embedding_a]
        if hparams.encode_t:
            self.embedding_t = torch.nn.Embedding(hparams.N_vocab, hparams.N_tau)
            self.embeddings['t'] = self.embedding_t
            self.models_to_train += [self.embedding_t]

        self.nerf_coarse = NeRF('coarse',
                                in_channels_xyz=6*hparams.N_emb_xyz+3,
                                in_channels_dir=6*hparams.N_emb_dir+3)
        self.models = {'coarse': self.nerf_coarse}
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF('fine',
                                  in_channels_xyz=6*hparams.N_emb_xyz+3,
                                  in_channels_dir=6*hparams.N_emb_dir+3,
                                  encode_appearance=hparams.encode_a,
                                  in_channels_a=hparams.N_a,
                                  encode_transient=hparams.encode_t,
                                  in_channels_t=hparams.N_tau,
                                  beta_min=hparams.beta_min)
            self.models['fine'] = self.nerf_fine
        self.models_to_train += [self.models]
        self.setting = hparams.setting

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, rays, ts, eval=False, avg_embedding=None):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in tqdm(list(range(0, B, self.hparams.chunk)), disable=B <= self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            ts[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back,
                            a_embedded=avg_embedding,
                            test_time=eval,
                            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v.cpu() if eval else v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir}
        if self.hparams.dataset_name == 'phototourism':
            kwargs['img_downscale'] = self.hparams.img_downscale
            kwargs['val_num'] = self.hparams.num_gpus
            kwargs['use_cache'] = self.hparams.use_cache
            kwargs['setting'] = self.setting
        elif self.hparams.dataset_name == 'blender':
            kwargs['img_wh'] = tuple(self.hparams.img_wh)
            kwargs['perturbation'] = self.hparams.data_perturb
        self.train_dataset = dataset(split='train', **kwargs)
        kwargs.pop("use_cache")
        self.val_dataset: PhototourismDataset = dataset(split='val', use_cache=False, **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models_to_train)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        rays, rgbs, ts = batch['rays'], batch['rgbs'], batch['ts']
        results = self(rays, ts)
        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)
        for k, v in loss_d.items():
            self.log(f'train/{k}', v, prog_bar=True)
        self.log('train/psnr', psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        # render 1st val image but skip others
        if self.current_epoch % 5 != 0 and batch_nb != 0:
            return

        rays, rgbs, ts = batch['rays'], batch['rgbs'], batch['ts']
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        ts = ts.squeeze() # (H*W)
        WH = batch['img_wh']

        # filename = self.val_dataset.image_paths[ts[0].item()]
        # sky_mask = np.array(Image.open(Path("/home/dawars/projects/master_thesis/NeuralRecon-W/output") / f"{filename}_mask.png"))

        # rgbs = rgbs.detach().clone()
        # del batch
        avg_embedding = self.embedding_a(torch.arange(0, self.val_dataset.N_images_train, device="cuda")).mean(0)

        results = self.forward(rays, ts, eval=True, avg_embedding=avg_embedding)
        # for k in ['weights_coarse', 'opacity_coarse', 'rgb_coarse', 'transient_sigmas', 'beta', 'rgb_fine', 'depth_fine',
        #           '_rgb_fine_static']:
        #     results[k] = results[k].to(self.device)
        # loss_d = self.loss(results, rgbs)
        # loss = sum(l for l in loss_d.values())
        log = {}
        # log = {'val_loss': loss}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        if self.trainer.global_rank == 0:
            if self.hparams.dataset_name == 'phototourism':
                W, H = WH[0, 0].item(), WH[0, 1].item()
            else:
                W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            # img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = results[f'depth_{typ}'].view(H, W) # (3, H, W)
            img_pred_static = results['rgb_fine_static'].view(H, W, 3).permute(2, 0, 1).cpu().numpy()
            _img_pred_static = results['_rgb_fine_static'].view(H, W, 3).permute(2, 0, 1).cpu().numpy()
            img_pred_transient = results['rgb_fine_transient'].view(H, W, 3).permute(2, 0, 1).cpu().numpy()
            depth_pred_static = results['depth_fine_static'].view(H, W)
            depth_pred_transient = results['depth_fine_transient'].view(H, W)

            # normal = visualize_normal(results[f'normal'].view(H, W)) # (3, H, W)

            if batch_nb == 0:
                from inference import apply_depth_colormap
                # near_plane = depth[sky_mask].min() - 1e-5
                # far_plane = depth[sky_mask].max() + 1e-5
                # depth_mask = apply_depth_colormap(depth, near_plane=near_plane, far_plane=far_plane)
                # static_depth = depth_pred_static.cpu().numpy()
                # near_plane = static_depth[sky_mask].min() - 1e-5
                # far_plane = static_depth[sky_mask].max() + 1e-5
                # depth_static_mask = apply_depth_colormap(static_depth, near_plane=near_plane, far_plane=far_plane)

                self.logger.log_image(key="Eval Images/img", images=[img], step=self.current_epoch)
                self.logger.log_image(key="Eval Images/img_transient", images=[img_pred_transient.transpose(1, 2, 0)], step=self.current_epoch)
                self.logger.log_image(key="Eval Images/img_static", images=[img_pred_static.transpose(1, 2, 0)], step=self.current_epoch)
                self.logger.log_image(key="Eval Images/_img_static", images=[_img_pred_static.transpose(1, 2, 0)], step=self.current_epoch)
                self.logger.log_image(key="Eval Images/depth", images=[visualize_depth(depth)], step=self.current_epoch)
                # self.logger.log_image(key="Eval Images/depth_mask", images=[depth_mask], step=self.current_epoch)
                self.logger.log_image(key="Eval Images/depth_static", images=[visualize_depth(depth_pred_static)], step=self.current_epoch)
                # self.logger.log_image(key="Eval Images/depth_static_mask", images=[depth_static_mask], step=self.current_epoch)
                self.logger.log_image(key="Eval Images/depth_transient", images=[visualize_depth(depth_pred_transient)], step=self.current_epoch)
                # self.logger.log_image(key="Eval Images/normal", images=[normal], step=self.current_epoch)

                # self.logger.experiment.add_images('val/img', img[None], self.global_step)
                # self.logger.experiment.add_images('val/img_static', img_pred_static[None], self.global_step)
                # self.logger.experiment.add_images('val/_img_static', _img_pred_static[None], self.global_step)
                # self.logger.experiment.add_images('val/depth_static', visualize_depth(depth_pred_static)[None], self.global_step)

                # plt.imsave(os.path.join(self.logger.root_dir, f"{filename}_rgb.png"), img_pred_static)
                # plt.imsave(os.path.join(self.logger.root_dir, f"{filename}_depth.png"),
                #            apply_depth_colormap(static_depth, near_plane=near_plane, far_plane=far_plane))
                # self.logger.experiment.add_images('val/depth_static_mask', depth_mask_2, self.global_step)
                # self.logger.experiment.add_images('val/depth_static_mask_white', apply_depth_colormap(static_depth, near_plane=near_plane, far_plane=far_plane, accumulation=sky_mask[..., np.newaxis]).transpose(2,0,1)[None], self.global_step)

        psnr_ = psnr(results[f'rgb_{typ}'], rgbs.cpu())
        psnr_static_ = psnr(results['rgb_fine_static'], rgbs.cpu())
        # psnr_mask = psnr(results[f'rgb_{typ}'], rgbs.cpu(), valid_mask=sky_mask.flatten())
        # psnr_static_mask = psnr(results['rgb_fine_static'], rgbs.cpu(), valid_mask=sky_mask.flatten())
        log['val_psnr'] = psnr_
        log['val_psnr_static'] = psnr_static_
        # log['val_psnr_mask'] = psnr_mask
        # log['val_psnr_static_mask'] = psnr_static_mask

        self.log('val/psnr', psnr_, on_epoch=True)
        self.log('val/psnr_static', psnr_static_, on_epoch=True)
        # self.log('val/psnr_mask', psnr_mask, on_epoch=True)
        # self.log('val/psnr_static_mask', psnr_static_mask, on_epoch=True)

    def on_validation_epoch_end(self, outputs):
        # mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        mean_psnr_static = torch.stack([x['val_psnr_static'] for x in outputs]).mean()
        # mean_psnr_mask = torch.stack([x['val_psnr_mask'] for x in outputs]).mean()
        # mean_psnr_static_mask = torch.stack([x['val_psnr_static_mask'] for x in outputs]).mean()

        # self.log('val/loss', mean_loss, sync_dist=True)
        self.log('val/psnr', mean_psnr, prog_bar=True, sync_dist=True)
        self.log('val/psnr_static', mean_psnr_static, prog_bar=True, sync_dist=True)
        # self.log('val/psnr_mask', mean_psnr_mask, prog_bar=True, sync_dist=True)
        # self.log('val/psnr_static_mask', mean_psnr_static_mask, prog_bar=True, sync_dist=True)


def main(hparams):
    # torch.set_float32_matmul_precision("high")
    system = NeRFSystem(hparams)
    exp_name = f"{hparams.exp_name}_{hparams.setting}"
    checkpoint_callback = \
        ModelCheckpoint(dirpath=os.path.join(hparams.save_path, 'ckpts', exp_name),
                        filename='{epoch:d}',
                        monitor='val/psnr',
                        mode='max',
                        save_top_k=-1)

    logger = WandbLogger(name=exp_name, dir=hparams.save_path, project="nerfw")

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      # strategy=DDPStrategy(find_unused_parameters=False),
                      callbacks=[checkpoint_callback],
                      logger=logger,
                      accelerator="gpu",
                      devices=hparams.num_gpus,
                      num_sanity_val_steps=1,
                      # limit_val_batches=2,
                      # check_val_every_n_epoch=5,
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus==1 else None)

    trainer.fit(system, ckpt_path=hparams.ckpt_path)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)
