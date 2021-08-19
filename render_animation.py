import os
from glob import glob
import numpy as np
import cv2
import torch
from derender import utils, rendering


EPS = 1e-7


def load_imgs(flist):
    return torch.stack([torch.FloatTensor(cv2.imread(f) /255.).flip(2) for f in flist], 0).permute(0,3,1,2)


def load_txts(flist):
    return torch.stack([torch.FloatTensor(np.loadtxt(f, delimiter=',')) for f in flist], 0)


def render_views(renderer, cam_loc, canon_sor_vtx, sor_faces, albedo, env_map, spec_alpha, spec_albedo, tx_size):
    b = canon_sor_vtx.size(0)
    s = 80
    rxs = torch.linspace(0, np.pi/3, s//2)
    rxs = torch.cat([rxs, rxs.flip(0)], 0)
    rys = torch.linspace(0, 2*np.pi, s)
    ims = []
    for i, (rx, ry) in enumerate(zip(rxs, rys)):
        rxyz = torch.stack([rx*0, ry, rx*0], 0).unsqueeze(0).to(canon_sor_vtx.device)
        sor_vtx = rendering.transform_pts(canon_sor_vtx, rxyz, None)
        rxyz = torch.stack([rx, ry*0, rx*0], 0).unsqueeze(0).to(canon_sor_vtx.device)
        sor_vtx = rendering.transform_pts(sor_vtx, rxyz, None)
        sor_vtx_map = rendering.get_sor_quad_center_vtx(sor_vtx)  # Bx(H-1)xTx3
        normal_map = rendering.get_sor_quad_center_normal(sor_vtx)  # Bx(H-1)xTx3
        diffuse, specular = rendering.envmap_phong_shading(sor_vtx_map, normal_map, cam_loc, env_map, spec_alpha)
        tex_im = rendering.compose_shading(albedo, diffuse, spec_albedo.view(b,1,1,1), specular).clamp(0,1)

        im_rendered = rendering.render_sor(renderer, sor_vtx, sor_faces.repeat(b,1,1,1,1), tex_im, tx_size=tx_size, dim_inside=True).clamp(0, 1)
        ims += [im_rendered]
    ims = torch.stack(ims, 1)  # BxTxCxHxW
    return ims


def render_relight(renderer, cam_loc, sor_vtx, sor_vtx_map, sor_faces, normal_map, albedo, spec_alpha, spec_albedo, tx_size):
    b = sor_vtx.size(0)
    lam = 20
    F = 0.15
    env_amb = 0.015
    n_sgls = 1
    sgl_lams = torch.FloatTensor([lam]).repeat(b, n_sgls).to(sor_vtx.device)
    sgl_Fs = torch.FloatTensor([F]).repeat(b, n_sgls).to(sor_vtx.device) *sgl_lams**0.5

    s = 80
    azims = torch.linspace(0, 4*np.pi, s)
    elevs = torch.linspace(0, np.pi/2, s//2)
    elevs = torch.cat([elevs, elevs.flip(0)], 0)

    ims = []
    for i, (azim, elev) in enumerate(zip(azims, elevs)):
        dy = -elev.sin()
        dx = elev.cos() * azim.sin()
        dz = -elev.cos() * azim.cos()
        sgl_dirs = torch.stack([dx, dy, dz], 0).repeat(b, n_sgls, 1).to(sor_vtx.device)
        sg_lights = torch.cat([sgl_dirs, sgl_lams.unsqueeze(2), sgl_Fs.unsqueeze(2)], 2).to(sor_vtx.device)

        env_map = rendering.sg_to_env_map(sg_lights, n_elev=16, n_azim=48)
        env_map_ambient = torch.FloatTensor([env_amb]).repeat(b).to(sor_vtx.device)
        env_map = env_map + env_map_ambient.view(b,1,1)

        diffuse, specular = rendering.envmap_phong_shading(sor_vtx_map, normal_map, cam_loc, env_map, spec_alpha)
        tex_im = rendering.compose_shading(albedo, diffuse, spec_albedo.view(b,1,1,1), specular).clamp(0,1)

        im_rendered = rendering.render_sor(renderer, sor_vtx, sor_faces.repeat(b,1,1,1,1), tex_im, tx_size=tx_size, dim_inside=True).clamp(0, 1)
        ims += [im_rendered]
    ims = torch.stack(ims, 1)  # BxTxCxHxW
    return ims


def main(in_dir, out_dir):
    device = 'cuda:0'
    image_size = 256
    radcol_height = 32
    sor_circum = 96
    tex_im_h = 256
    tex_im_w = 768
    env_map_h = 16
    env_map_w = 48
    fov = 10  # in degrees
    ori_z = 5
    tx_size = 8
    cam_loc = torch.FloatTensor([0,0,-ori_z]).to(device)
    sor_faces = rendering.get_sor_full_face_idx(radcol_height, sor_circum).to(device)  # 2x(H-1)xWx3
    renderer = rendering.get_renderer(world_ori=[0,0,ori_z], image_size=image_size, fov=fov, fill_back=True, device='cuda:0')
    batch_size = 10

    sor_curve_all = load_txts(sorted(glob(os.path.join(in_dir, 'sor_curve/*_sor_curve.txt'), recursive=True)))
    albedo_all = load_imgs(sorted(glob(os.path.join(in_dir, 'albedo_map/*_albedo_map.png'), recursive=True)))
    mask_gt_all = load_imgs(sorted(glob(os.path.join(in_dir, 'mask_gt/*_mask_gt.png'), recursive=True)))
    pose_all = load_txts(sorted(glob(os.path.join(in_dir, 'pose/*_pose.txt'), recursive=True)))
    material_all = load_txts(sorted(glob(os.path.join(in_dir, 'material/*_material.txt'), recursive=True)))
    env_map_all = load_imgs(sorted(glob(os.path.join(in_dir, 'env_map/*_env_map.png'), recursive=True)))[:,0,:,:]

    total_num = sor_curve_all.size(0)
    for b0 in range(0, total_num, batch_size):
        b1 = min(total_num, b0+batch_size)
        b = b1 - b0
        print("Rendering %d-%d/%d" %(b0, b1, total_num))

        sor_curve = sor_curve_all[b0:b1].to(device)
        albedo = albedo_all[b0:b1].to(device)
        mask_gt = mask_gt_all[b0:b1].to(device)
        pose = pose_all[b0:b1].to(device)
        material = material_all[b0:b1].to(device)
        env_map = env_map_all[b0:b1].to(device)

        canon_sor_vtx = rendering.get_sor_vtx(sor_curve, sor_circum)  # BxHxTx3
        rxyz = pose[:,:3] /180 *np.pi
        txy = pose[:,3:]
        sor_vtx = rendering.transform_pts(canon_sor_vtx, rxyz, txy)
        sor_vtx_map = rendering.get_sor_quad_center_vtx(sor_vtx)  # Bx(H-1)xTx3
        normal_map = rendering.get_sor_quad_center_normal(sor_vtx)  # Bx(H-1)xTx3
        spec_alpha, spec_albedo = material.unbind(1)

        ## replicate albedo
        wcrop_ratio = 1/6
        wcrop_tex_im = int(wcrop_ratio *tex_im_w//2)
        albedo = rendering.gamma(albedo)
        p = 8
        front_albedo = torch.cat([albedo[:,:,:,p:2*p].flip(3), albedo[:,:,:,p:-p], albedo[:,:,:,-2*p:-p].flip(3)], 3)
        albedo_replicated = torch.cat([front_albedo[:,:,:,:wcrop_tex_im].flip(3), front_albedo, front_albedo.flip(3), front_albedo[:,:,:,:-wcrop_tex_im]], 3)

        with torch.no_grad():
            novel_views = render_views(renderer, cam_loc, canon_sor_vtx, sor_faces, albedo_replicated, env_map, spec_alpha, spec_albedo, tx_size)
            relightings = render_relight(renderer, cam_loc, sor_vtx, sor_vtx_map, sor_faces, normal_map, albedo_replicated, spec_alpha, spec_albedo, tx_size)
            [utils.save_images(out_dir, novel_views[:,i].cpu().numpy(), suffix='novel_views_%d'%i, sep_folder=True) for i in range(0, novel_views.size(1), relightings.size(1)//10)]
            utils.save_videos(out_dir, novel_views.cpu().numpy(), suffix='novel_view_videos', sep_folder=True, fps=25)
            [utils.save_images(out_dir, relightings[:,i].cpu().numpy(), suffix='relight_%d'%i, sep_folder=True) for i in range(0, relightings.size(1), relightings.size(1)//10)]
            utils.save_videos(out_dir, relightings.cpu().numpy(), suffix='relight_videos', sep_folder=True, fps=25)


if __name__ == '__main__':
    in_dir = 'results/met_vase_pretrained/test_results_checkpoint500'
    out_dir = 'results/met_vase_pretrained/test_results_checkpoint500/animations'
    main(in_dir, out_dir)
