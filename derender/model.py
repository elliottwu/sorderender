import os
import math
import glob
import torch
import torch.nn as nn
import torchvision
from .networks import SoRNet, EnvMapNet, DiscNet
from .GAN import UnetGenerator, GANLoss, set_requires_grad
from . import utils, rendering


class Derenderer():
    def __init__(self, cfgs):
        self.model_name = cfgs.get('model_name', self.__class__.__name__)
        self.device = cfgs.get('device', 'cpu')
        self.image_size = cfgs.get('image_size', 256)
        self.wcrop_ratio = 1/6  # ratio of width to be cropped out for perspective projection
        self.radcol_height = cfgs.get('radcol_height', 32)
        self.sor_circum = cfgs.get('sor_circum', 96)
        self.tex_im_h = cfgs.get('tex_im_h', 256)
        self.tex_im_w = cfgs.get('tex_im_w', 768)
        self.env_map_h = cfgs.get('env_map_h', 32)
        self.env_map_w = cfgs.get('env_map_w', 64)
        self.fov = cfgs.get('fov', 10)
        self.ori_z = cfgs.get('ori_z', 5)
        self.x_rotation_min = cfgs.get('x_rotation_min', -10)
        self.x_rotation_max = cfgs.get('x_rotation_max', 10)
        self.z_rotation_min = cfgs.get('z_rotation_min', -10)
        self.z_rotation_max = cfgs.get('z_rotation_max', 10)
        self.xy_translation_range = cfgs.get('xy_translation_range', 0.2)
        self.max_range = math.tan(self.fov/2/180*math.pi) *self.ori_z
        self.min_depth = self.ori_z - self.max_range
        self.max_depth = self.ori_z + self.max_range
        self.min_hscale = cfgs.get('min_hscale', 0.5)
        self.max_hscale = cfgs.get('max_hscale', 0.9)
        self.spec_albedo_min = cfgs.get('spec_albedo_min', 0.)
        self.spec_albedo_max = cfgs.get('spec_albedo_max', 1.)
        self.tx_size = cfgs.get('tx_size', 4)
        self.renderer = rendering.get_renderer(world_ori=[0, 0, self.ori_z], image_size=self.image_size, fov=self.fov, fill_back=True, device=self.device)

        ## losses
        self.lam_mask = cfgs.get('lam_mask', 1)
        self.lam_mask_dt = cfgs.get('lam_mask_dt', 1)
        self.lam_im = cfgs.get('lam_im', 1)
        self.lam_mean_albedo_im = cfgs.get('lam_mean_albedo_im', 0)
        self.diffuse_reg_mean = cfgs.get('diffuse_reg_mean', 0.5)
        self.diffuse_reg_margin = cfgs.get('diffuse_reg_margin', 0.1)
        self.lam_diffuse_reg = cfgs.get('lam_diffuse_reg', 0)
        self.lam_GAN = cfgs.get('lam_GAN', 0)
        self.sample_patch_size = cfgs.get('sample_patch_size', 64)

        ## networks and optimizers
        self.lr = cfgs.get('lr', 1e-4)
        self.netR = SoRNet(cin=3, cout2=5, in_size=self.image_size, out_size=self.radcol_height, zdim=128, nf=64, activation=nn.Sigmoid)  # radius column, height scale, rx, rz, tx, ty
        self.netT = UnetGenerator(input_nc=3, output_nc=3, num_downs=6, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, f_act=nn.Tanh)
        self.netE = EnvMapNet(cin=6, cout=1, cout2=2, in_size=self.image_size, out_size=self.env_map_h, nf=64, zdim=128, activation=nn.Sigmoid)  # spec_alpha, spec_albedo
        self.netD = DiscNet(cin=3, cout=1, nf=64, norm=nn.InstanceNorm2d, activation=None)
        self.criterionGAN = GANLoss('lsgan')
        self.network_names = [k for k in vars(self) if k.startswith('net')]
        self.make_optimizer = lambda model, lr=self.lr: torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, betas=(0.9, 0.999))

        ## other parameters
        self.sor_faces = rendering.get_sor_full_face_idx(self.radcol_height, self.sor_circum)  # 2x(H-1)xWx3
        self.other_param_names = ['sor_faces', 'criterionGAN']

        ## range rescalers
        self.rad_col_rescaler = lambda x : x *0.9 +0.05
        self.hscale_rescaler = lambda x : x *(self.max_hscale-self.min_hscale) + self.min_hscale
        self.rx_rescaler = lambda x : (x * (self.x_rotation_max-self.x_rotation_min) + self.x_rotation_min) /180*math.pi
        self.rz_rescaler = lambda x : (x * (self.z_rotation_max-self.z_rotation_min) + self.z_rotation_min) /180*math.pi
        self.txy_rescaler = lambda x : (x*2 -1) *self.max_range*self.xy_translation_range
        self.spec_alpha_rescaler = lambda x : (x *13 +1)**2  # 1~196
        self.spec_albedo_rescaler = lambda x : x *(self.spec_albedo_max-self.spec_albedo_min) + self.spec_albedo_min

    def init_optimizers(self):
        self.optimizer_names = []
        for net_name in self.network_names:
            optimizer = self.make_optimizer(getattr(self, net_name))
            optim_name = net_name.replace('net','optimizer')
            setattr(self, optim_name, optimizer)
            self.optimizer_names += [optim_name]

    def load_model_state(self, cp):
        for k in cp:
            if k and k in self.network_names:
                getattr(self, k).load_state_dict(cp[k])

    def load_optimizer_state(self, cp):
        for k in cp:
            if k and k in self.optimizer_names:
                getattr(self, k).load_state_dict(cp[k])

    def get_model_state(self):
        states = {}
        for net_name in self.network_names:
            states[net_name] = getattr(self, net_name).state_dict()
        return states

    def get_optimizer_state(self):
        states = {}
        for optim_name in self.optimizer_names:
            states[optim_name] = getattr(self, optim_name).state_dict()
        return states

    def to_device(self, device):
        self.device = device
        for net_name in self.network_names:
            setattr(self, net_name, nn.DataParallel(getattr(self, net_name).to(device)))
        if hasattr(self, 'other_param_names'):
            for param_name in self.other_param_names:
                setattr(self, param_name, getattr(self, param_name).to(device))

    def set_train(self):
        for net_name in self.network_names:
            getattr(self, net_name).train()

    def set_eval(self):
        for net_name in self.network_names:
            getattr(self, net_name).eval()

    def get_real_fake_patches(self):
        b = self.albedo.size(0)
        num_patch = 8
        num_patch_each = int(b*num_patch *0.25)
        ps = self.sample_patch_size
        wcrop_tex_im = int(self.wcrop_ratio *self.tex_im_w//2)
        albedo_cropped = self.albedo[:,:,:,wcrop_tex_im:self.tex_im_w//2-wcrop_tex_im]
        specular_map = (self.specular *self.spec_albedo)
        specular_map = torch.nn.functional.interpolate(specular_map, (self.tex_im_h, self.tex_im_w), mode='bilinear', align_corners=False)
        specular_map_cropped = specular_map[:,:,:,wcrop_tex_im:self.tex_im_w//2-wcrop_tex_im]
        albedo_specular_cat = torch.cat([albedo_cropped, specular_map_cropped], 1)
        patches = utils.get_patches(albedo_specular_cat, num_patch=num_patch, patch_size=ps, scale=(0.25,0.5))  # BxNxCxHxW
        albedo_patches = patches[:,:,:3].reshape(b*num_patch, 3, ps, ps)
        specular_patches = patches[:,:,3:4].reshape(b*num_patch, 1, ps, ps)

        ## sort by specularity variance
        sort_id = specular_patches.view(b*num_patch, -1).var(1).argsort(0, descending=False)

        ## shuffle top 50%
        sort_id_1, sort_id_2 = sort_id.chunk(2, 0)
        sort_id_2_shuffled = sort_id_2[torch.randperm(len(sort_id_2))]
        sort_id = torch.cat([sort_id_1, sort_id_2_shuffled], 0)

        albedo_patches_sorted = albedo_patches[sort_id]
        self.nonspec_albedo_patches = albedo_patches_sorted[:num_patch_each]
        self.spec_albedo_patches = albedo_patches_sorted[-num_patch_each:]
        specular_patches_sorted = specular_patches[sort_id]
        self.nonspec_spec_patches = specular_patches_sorted[:num_patch_each]
        self.spec_spec_patches = specular_patches_sorted[-num_patch_each:]
        return self.nonspec_albedo_patches, self.spec_albedo_patches

    def backward(self):
        real_patches, fake_patches = self.get_real_fake_patches()

        ## backward G
        set_requires_grad(self.netD, False)
        for optim_name in self.optimizer_names:
            if optim_name == 'optimizerD':
                continue
            getattr(self, optim_name).zero_grad()
        self.GAN_G_pred = self.netD(fake_patches *2-1)
        self.loss_G_GAN = self.criterionGAN(self.GAN_G_pred, True)
        loss_G_total = self.loss_total + self.lam_GAN*self.loss_G_GAN
        loss_G_total.backward()
        for optim_name in self.optimizer_names:
            if optim_name == 'optimizerD':
                continue
            getattr(self, optim_name).step()

        ## backward D
        set_requires_grad(self.netD, True)
        self.optimizerD.zero_grad()
        self.GAN_D_real_pred = self.netD(real_patches.detach() *2-1)
        self.GAN_D_fake_pred = self.netD(fake_patches.detach() *2-1)
        self.loss_D_GAN_real = self.criterionGAN(self.GAN_D_real_pred, True)
        self.loss_D_GAN_fake = self.criterionGAN(self.GAN_D_fake_pred, False)
        loss_D_total = (self.loss_D_GAN_real + self.loss_D_GAN_fake) * 0.5
        loss_D_total.backward()
        self.optimizerD.step()

    def forward(self, input):
        if isinstance(input, tuple) or isinstance(input, list):
            input_im, mask_gt, mask_gt_dt = input
            mask_gt = mask_gt[:,0,:,:]
            mask_gt_dt = mask_gt_dt / self.image_size
        else:
            input_im = input
            mask_gt = input_im[:,0,:,:] *0+1
            mask_gt_dt = input_im[:,0,:,:] *0
        self.input_im = input_im.to(self.device)
        self.mask_gt = mask_gt.to(self.device)
        self.mask_gt_dt = mask_gt_dt.to(self.device)
        b, c, h, w = self.input_im.shape

        ## predict sor and pose
        self.rad_col, pose_pred = self.netR(self.input_im *2-1)  # BxH, 0~1
        self.rad_col = self.rad_col_rescaler(self.rad_col)
        self.hscale, self.rx, self.rz, self.tx, self.ty = pose_pred.unbind(1)  # 0~1
        self.hscale = self.hscale_rescaler(self.hscale)
        self.sor_curve = rendering.get_sor_curve(self.rad_col, self.hscale) *self.max_range

        self.rx = self.rx_rescaler(self.rx)
        self.rz = self.rz_rescaler(self.rz)
        self.rxyz = torch.stack([self.rx, torch.zeros_like(self.rx), self.rz], 1)
        self.txy = torch.stack([self.tx, self.ty], 1)
        self.txy = self.txy_rescaler(self.txy)

        self.canon_sor_vtx = rendering.get_sor_vtx(self.sor_curve, self.sor_circum)  # BxHxTx3
        self.sor_vtx = rendering.transform_pts(self.canon_sor_vtx, self.rxyz, self.txy)
        self.normal_map = rendering.get_sor_quad_center_normal(self.sor_vtx)  # Bx(H-1)xTx3

        ## render mask
        self.mask_rendered = self.renderer.render_silhouettes(self.sor_vtx.view(b,-1,3), self.sor_faces.view(1,-1,3).repeat(b,1,1))

        ## sample frontal texture map
        self.sor_vtx_map = rendering.get_sor_quad_center_vtx(self.sor_vtx)  # Bx(H-1)xTx3
        wcrop_tex_im = int(self.wcrop_ratio *self.tex_im_w//2)
        self.front_tex_im_gt = rendering.unwrap_sor_front_tex_im(self.sor_vtx_map[:,:,:self.sor_circum//2,:], self.tex_im_h, self.tex_im_w//2, self.renderer.K, [0, 0, self.ori_z], self.input_im)

        ## predict albedo
        self.front_tex_im_gt_cropped = self.front_tex_im_gt[:,:,:,wcrop_tex_im:-wcrop_tex_im]  # 128x128
        self.albedo = self.netT(self.front_tex_im_gt_cropped *2-1) /2+0.5
        self.albedo = torch.nn.functional.pad(self.albedo, (wcrop_tex_im,wcrop_tex_im,0,0), mode='replicate')

        ## replicate back side
        self.albedo = torch.cat([self.albedo, self.albedo.flip(3)], 3)

        ## shading
        wcrop_sor = int(self.wcrop_ratio*self.sor_circum//2)
        normal_map_cropped = self.normal_map[:,:,wcrop_sor:self.sor_circum//2-wcrop_sor,:]
        normal_map_cropped = torch.nn.functional.interpolate(normal_map_cropped.permute(0,3,1,2), (self.tex_im_h, self.tex_im_w//2-wcrop_tex_im*2), mode='bilinear', align_corners=False)
        normal_map_cropped = normal_map_cropped / (normal_map_cropped**2).sum(1,keepdim=True)**0.5
        light_input = torch.cat([self.front_tex_im_gt_cropped *2-1, normal_map_cropped], 1)
        self.env_map, self.light_params = self.netE(light_input)
        self.env_map = self.env_map.squeeze(1)  # 0~1
        self.spec_alpha = self.spec_alpha_rescaler(self.light_params[:,0])
        self.spec_albedo_scalar = self.spec_albedo_rescaler(self.light_params[:,1])
        self.spec_albedo = self.spec_albedo_scalar.view(b,1,1,1)
        cam_loc = torch.FloatTensor([0,0,-self.ori_z]).to(self.device)
        self.diffuse, self.specular = rendering.envmap_phong_shading(self.sor_vtx_map, self.normal_map, cam_loc, self.env_map, self.spec_alpha)
        self.tex_im = rendering.compose_shading(self.albedo, self.diffuse, self.spec_albedo, self.specular).clamp(0,1)

        ## render
        self.front_tex_im = self.tex_im[:,:,:,wcrop_tex_im:self.tex_im_w//2-wcrop_tex_im]
        self.front_sor_faces = self.sor_faces[:,:,wcrop_sor:self.sor_circum//2-wcrop_sor].repeat(b,1,1,1,1)
        self.front_mask_rendered = self.renderer.render_silhouettes(self.sor_vtx.view(b,-1,3), self.front_sor_faces.reshape(b,-1,3))
        self.front_im_rendered = rendering.render_sor(self.renderer, self.sor_vtx, self.front_sor_faces, self.front_tex_im, tx_size=self.tx_size).clamp(0, 1)
        im_loss_mask = (self.front_mask_rendered * self.mask_gt).unsqueeze(1)
        im_loss_mask = im_loss_mask.expand_as(self.input_im).detach()

        ## render with mean albedo
        mean_albedo = self.albedo[:,:,:,wcrop_tex_im:self.tex_im_w//2-wcrop_tex_im].reshape(b,3,-1).mean(2)
        self.mean_albedo = mean_albedo.view(b,3,1,1).expand_as(self.albedo)
        self.mean_albedo_tex_im = rendering.compose_shading(self.mean_albedo, self.diffuse, self.spec_albedo, self.specular).clamp(0,1)
        self.mean_albedo_front_tex_im = self.mean_albedo_tex_im[:,:,:,wcrop_tex_im:self.tex_im_w//2-wcrop_tex_im]
        self.mean_albedo_front_im_rendered = rendering.render_sor(self.renderer, self.sor_vtx, self.front_sor_faces, self.mean_albedo_front_tex_im, tx_size=self.tx_size).clamp(0, 1)

        ## losses
        self.loss_mask = ((self.mask_rendered - self.mask_gt)**2).view(b,-1).mean(1).mean()
        self.loss_mask_dt = (self.mask_rendered * self.mask_gt_dt).view(b,-1).mean(1).mean()
        self.loss_im = ((((self.front_im_rendered - self.input_im).abs()) *im_loss_mask).reshape(b,-1).sum(1) / im_loss_mask.reshape(b,-1).sum(1)).mean()
        self.loss_mean_albedo_im = ((((self.mean_albedo_front_im_rendered - self.input_im).abs()) *im_loss_mask).reshape(b,-1).sum(1) / im_loss_mask.reshape(b,-1).sum(1)).mean()

        ## diffuse consistency regularizer
        self.diffuse_mean = self.diffuse[:,:,:,wcrop_sor:self.sor_circum//2-wcrop_sor].reshape(b,-1).mean(1)
        self.loss_diffuse_reg = (((self.diffuse_mean - self.diffuse_reg_mean).abs() -self.diffuse_reg_margin).clamp(min=0.) **2).mean()

        ## total loss
        self.loss_total = self.lam_mask*self.loss_mask + self.lam_mask_dt*self.loss_mask_dt + self.lam_im*self.loss_im + self.lam_mean_albedo_im*self.loss_mean_albedo_im + self.lam_diffuse_reg*self.loss_diffuse_reg
        metrics = {'loss': self.loss_total}

        ## for visualization
        self.front_im_rendered_masked = self.front_im_rendered *im_loss_mask
        self.mean_albedo_front_im_rendered_masked = self.mean_albedo_front_im_rendered *im_loss_mask
        self.albedo = rendering.tonemap(self.albedo)
        self.mean_albedo = rendering.tonemap(self.mean_albedo)

        return metrics

    def visualize(self, logger, total_iter, max_bs=25):
        b, c, h, w = self.input_im.shape
        wcrop_tex_im = int(self.wcrop_ratio *self.tex_im_w//2)
        wcrop_sor = int(self.wcrop_ratio*self.sor_circum//2)

        self.depth_rendered = self.renderer.render_depth(self.sor_vtx.view(b,-1,3), self.sor_faces.view(1,-1,3).repeat(b,1,1)).clamp(self.min_depth, self.max_depth)
        self.normal_rendered = rendering.render_sor(self.renderer, self.sor_vtx, self.sor_faces.repeat(b,1,1,1,1), self.normal_map.permute(0,3,1,2), tx_size=self.tx_size, render_normal=True).clamp(0, 1)
        self.im_rendered = rendering.render_sor(self.renderer, self.sor_vtx, self.sor_faces.repeat(b,1,1,1,1), self.tex_im, tx_size=self.tx_size, dim_inside=True).clamp(0, 1)
        self.albedo_rendered = rendering.render_sor(self.renderer, self.sor_vtx, self.sor_faces.repeat(b,1,1,1,1), self.albedo, tx_size=self.tx_size, dim_inside=True).clamp(0, 1)
        self.diffuse_map = self.diffuse.clamp(0, 1).repeat(1,3,1,1)
        self.diffuse_rendered = rendering.render_sor(self.renderer, self.sor_vtx, self.sor_faces.repeat(b,1,1,1,1), self.diffuse_map, tx_size=self.tx_size).clamp(0, 1)
        self.specular_map = (self.specular *self.spec_albedo).clamp(0, 1).repeat(1,3,1,1)
        self.specular_rendered = rendering.render_sor(self.renderer, self.sor_vtx, self.sor_faces.repeat(b,1,1,1,1), self.specular_map, tx_size=self.tx_size).clamp(0, 1)

        def log_grid_image(label, im, iter=total_iter):
            b = min(max_bs, im.size(0))
            nrow = int(math.ceil(b**0.5))
            im_grid = torchvision.utils.make_grid(im[:b].detach().cpu(), nrow=nrow)
            logger.add_image(label, im_grid, iter)

        log_grid_image('Mask/mask_gt', self.mask_gt.unsqueeze(1))
        log_grid_image('Mask/mask_gt_dt', self.mask_gt_dt.unsqueeze(1))
        log_grid_image('Mask/mask_rendered', self.mask_rendered.unsqueeze(1))
        log_grid_image('Mask/front_mask_rendered', self.front_mask_rendered.unsqueeze(1))

        log_grid_image('Rendered/input_image', self.input_im)
        log_grid_image('Rendered/im_rendered', self.im_rendered)
        log_grid_image('Rendered/depth_rendered', ((self.depth_rendered -self.min_depth)/(self.max_depth-self.min_depth)).unsqueeze(1))
        log_grid_image('Rendered/normal_rendered', self.normal_rendered)
        log_grid_image('Rendered/im_rendered_masked', self.front_im_rendered_masked)
        log_grid_image('Rendered/mean_albedo_im_rendered_masked', self.mean_albedo_front_im_rendered_masked)
        log_grid_image('Rendered/albedo_rendered', self.albedo_rendered)
        log_grid_image('Rendered/diffuse_rendered', self.diffuse_rendered)
        log_grid_image('Rendered/specular_rendered', self.specular_rendered)

        log_grid_image('Unwrap/albedo_map', self.albedo[:,:,:,wcrop_tex_im:self.tex_im_w//2-wcrop_tex_im])
        log_grid_image('Unwrap/mean_albedo_map', self.mean_albedo[:,:,:,wcrop_tex_im:self.tex_im_w//2-wcrop_tex_im])
        log_grid_image('Unwrap/normal_map', self.normal_map[:,:,wcrop_sor:self.sor_circum//2-wcrop_sor,:].permute(0,3,1,2) /2+0.5)
        log_grid_image('Unwrap/diffuse_map', self.diffuse_map[:,:,:,wcrop_sor:self.sor_circum//2-wcrop_sor])
        log_grid_image('Unwrap/specular_map', self.specular_map[:,:,:,wcrop_sor:self.sor_circum//2-wcrop_sor])
        log_grid_image('Unwrap/env_map', self.env_map.unsqueeze(1))
        log_grid_image('Unwrap/front_tex_im_gt', self.front_tex_im_gt[:,:,:,wcrop_tex_im:self.tex_im_w//2-wcrop_tex_im])
        log_grid_image('Unwrap/front_tex_im', self.tex_im[:,:,:,wcrop_tex_im:self.tex_im_w//2-wcrop_tex_im])
        log_grid_image('Unwrap/mean_albedo_front_tex_im', self.mean_albedo_tex_im[:,:,:,wcrop_tex_im:self.tex_im_w//2-wcrop_tex_im])

        logger.add_scalar('Loss/loss_total', self.loss_total, total_iter)
        logger.add_scalar('Loss/loss_mask', self.loss_mask, total_iter)
        logger.add_scalar('Loss/loss_mask_dt', self.loss_mask_dt, total_iter)
        logger.add_scalar('Loss/loss_im', self.loss_im, total_iter)
        logger.add_scalar('Loss/loss_mean_albedo_im', self.loss_mean_albedo_im, total_iter)
        logger.add_scalar('Loss/loss_diffuse_reg', self.loss_diffuse_reg, total_iter)

        logger.add_histogram('Pose/hscale', self.hscale.detach().cpu(), total_iter)
        pose = torch.cat([self.rxyz /math.pi*180, self.txy], 1).detach().cpu()
        vlist = ['pose_rx', 'pose_ry', 'pose_rz', 'pose_tx', 'pose_ty', 'pose_tz']
        for i in range(pose.shape[1]):
            logger.add_histogram('Pose/'+vlist[i], pose[:,i], total_iter)

        logger.add_histogram('Light/spec_alpha', self.spec_alpha.detach().cpu(), total_iter)
        logger.add_histogram('Light/spec_albedo_scalar', self.spec_albedo_scalar.detach().cpu(), total_iter)
        logger.add_histogram('Light/diffuse_mean', self.diffuse_mean.detach().cpu(), total_iter)
        logger.add_histogram('Light/env_map_hist', self.env_map.detach().cpu(), total_iter)

        logger.add_scalar('GAN/loss_G_GAN', self.loss_G_GAN, total_iter)
        logger.add_scalar('GAN/loss_D_GAN_real', self.loss_D_GAN_real, total_iter)
        logger.add_scalar('GAN/loss_D_GAN_fake', self.loss_D_GAN_fake, total_iter)
        log_grid_image('GAN/nonspec_albedo_patches', self.nonspec_albedo_patches)
        log_grid_image('GAN/spec_albedo_patches', self.spec_albedo_patches)
        log_grid_image('GAN/nonspec_spec_patches', self.nonspec_spec_patches.clamp(0,1).repeat(1,3,1,1))
        log_grid_image('GAN/spec_spec_patches', self.spec_spec_patches.clamp(0,1).repeat(1,3,1,1))
        logger.add_histogram('GAN/GAN_G_pred', self.GAN_G_pred.detach().cpu(), total_iter)
        logger.add_histogram('GAN/GAN_D_real_pred', self.GAN_D_real_pred.detach().cpu(), total_iter)
        logger.add_histogram('GAN/GAN_D_fake_pred', self.GAN_D_fake_pred.detach().cpu(), total_iter)

    def save_results(self, save_dir):
        b, c, h, w = self.input_im.shape
        wcrop_tex_im = int(self.wcrop_ratio *self.tex_im_w//2)
        wcrop_sor = int(self.wcrop_ratio*self.sor_circum//2)

        ## replicate textures
        front_albedo = rendering.gamma(self.albedo[:,:,:,wcrop_tex_im:self.tex_im_w//2-wcrop_tex_im])
        p = 8
        front_albedo = torch.cat([front_albedo[:,:,:,p:2*p].flip(3), front_albedo[:,:,:,p:-p], front_albedo[:,:,:,-2*p:-p].flip(3)], 3)
        albedo_replicated = torch.cat([front_albedo[:,:,:,:wcrop_tex_im].flip(3), front_albedo, front_albedo.flip(3), front_albedo[:,:,:,:-wcrop_tex_im]], 3)
        tex_im_replicated = rendering.compose_shading(albedo_replicated, self.diffuse, self.spec_albedo, self.specular).clamp(0,1)

        self.depth_rendered = self.renderer.render_depth(self.sor_vtx.view(b,-1,3), self.sor_faces.view(1,-1,3).repeat(b,1,1)).clamp(self.min_depth, self.max_depth)
        self.normal_rendered = rendering.render_sor(self.renderer, self.sor_vtx, self.sor_faces.repeat(b,1,1,1,1), self.normal_map.permute(0,3,1,2), tx_size=self.tx_size, render_normal=True).clamp(0, 1)
        self.im_rendered = rendering.render_sor(self.renderer, self.sor_vtx, self.sor_faces.repeat(b,1,1,1,1), tex_im_replicated, tx_size=self.tx_size, dim_inside=True).clamp(0, 1)
        self.albedo_rendered = rendering.render_sor(self.renderer, self.sor_vtx, self.sor_faces.repeat(b,1,1,1,1), rendering.tonemap(albedo_replicated), tx_size=self.tx_size, dim_inside=True).clamp(0, 1)
        self.mean_albedo_im_rendered = rendering.render_sor(self.renderer, self.sor_vtx, self.sor_faces.repeat(b,1,1,1,1), self.mean_albedo_tex_im, tx_size=self.tx_size, dim_inside=True).clamp(0, 1)
        self.diffuse_map = (self.diffuse *0.7).clamp(0, 1).repeat(1,3,1,1)
        self.diffuse_rendered = rendering.render_sor(self.renderer, self.sor_vtx, self.sor_faces.repeat(b,1,1,1,1), self.diffuse_map, tx_size=self.tx_size).clamp(0, 1)
        self.specular_map = (self.specular *self.spec_albedo *1.5).clamp(0, 1).repeat(1,3,1,1)
        self.specular_rendered = rendering.render_sor(self.renderer, self.sor_vtx, self.sor_faces.repeat(b,1,1,1,1), self.specular_map, tx_size=self.tx_size).clamp(0, 1)

        def save_images(im, suffix):
            utils.save_images(save_dir, im.detach().cpu().numpy(), suffix=suffix, sep_folder=True)
        def save_txt(data, suffix):
            utils.save_txt(save_dir, data.detach().cpu().numpy(), suffix=suffix, sep_folder=True)

        save_images(self.input_im, 'input_image')
        save_images(self.im_rendered, 'im_rendered')
        save_images(self.mask_gt.unsqueeze(1).repeat(1,3,1,1), 'mask_gt')
        save_images(self.front_mask_rendered.unsqueeze(1).repeat(1,3,1,1), 'front_mask_rendered')
        save_images(self.mask_rendered.unsqueeze(1).repeat(1,3,1,1), 'mask_rendered')
        save_images(((self.depth_rendered -self.min_depth)/(self.max_depth-self.min_depth)).unsqueeze(1).repeat(1,3,1,1), 'depth_rendered')
        save_images(self.normal_rendered, 'normal_rendered')
        save_images(self.albedo_rendered, 'albedo_rendered')
        save_images(self.diffuse_rendered, 'diffuse_rendered')
        save_images(self.specular_rendered, 'specular_rendered')
        save_images(self.mean_albedo_im_rendered, 'mean_albedo_im_rendered')
        save_images(self.albedo[:,:,:,wcrop_tex_im:self.tex_im_w//2-wcrop_tex_im], 'albedo_map')
        save_images(self.mean_albedo[:,:,:,wcrop_tex_im:self.tex_im_w//2-wcrop_tex_im], 'mean_albedo_map')
        save_images(self.normal_map.permute(0,3,1,2)[:,:,:,wcrop_sor:self.sor_circum//2-wcrop_sor] /2+0.5, 'normal_map')
        save_images(self.diffuse_map[:,:,:,wcrop_sor:self.sor_circum//2-wcrop_sor], 'diffuse_map')
        save_images(self.specular_map[:,:,:,wcrop_sor:self.sor_circum//2-wcrop_sor], 'specular_map')
        save_images(self.env_map.unsqueeze(1).repeat(1,3,1,1), 'env_map')
        save_images(self.front_tex_im_gt[:,:,:,wcrop_tex_im:self.tex_im_w//2-wcrop_tex_im], 'texture_map_gt')
        save_images(self.tex_im[:,:,:,wcrop_tex_im:self.tex_im_w//2-wcrop_tex_im], 'texture_map')
        save_images(self.mean_albedo_tex_im[:,:,:,wcrop_tex_im:self.tex_im_w//2-wcrop_tex_im], 'texture_map_mean_albedo')
        save_txt(self.sor_curve, suffix='sor_curve')
        save_txt(torch.cat([self.rxyz /math.pi*180, self.txy], 1), suffix='pose')
        save_txt(torch.stack([self.spec_alpha, self.spec_albedo_scalar], 1), suffix='material')

        ## other rendering
        material_rendered = rendering.render_material(self.renderer, self.spec_alpha, self.spec_albedo_scalar)
        save_images(material_rendered, 'material_rendered')
        novel_view_rendered, novel_view_mask_rendered = rendering.render_novel_view(self.renderer, self.canon_sor_vtx, self.sor_faces.repeat(b,1,1,1,1), albedo_replicated, self.spec_albedo, self.spec_alpha, self.env_map)
        save_images(novel_view_rendered, 'novel_view_rendered')
        save_images(novel_view_mask_rendered.unsqueeze(1).repeat(1,3,1,1), 'novel_view_mask_rendered')

    def save_scores(self, path):
        pass
