import math
import numpy as np
from PIL import Image
import cv2
import os
from glob import glob
import torch
import torchvision
import neural_renderer as nr


EPS = 1e-7


def get_renderer(world_ori=[0,0,1], image_size=128, fov=30, renderer_min_depth=0.1, renderer_max_depth=10, fill_back=True, device='cuda:0'):
    #### camera intrinsics
    #             (u)   (x)
    #    d * K^-1 (v) = (y)
    #             (1)   (z)

    ## renderer for visualization
    R = [[[1.,0.,0.],
          [0.,1.,0.],
          [0.,0.,1.]]]
    R = torch.FloatTensor(R).to(device)
    t = torch.FloatTensor(world_ori).to(device)
    fx = (image_size)/2/(math.tan(fov/2 *math.pi/180))
    fy = (image_size)/2/(math.tan(fov/2 *math.pi/180))
    cx = (image_size)/2
    cy = (image_size)/2
    K = [[fx, 0., cx],
         [0., fy, cy],
         [0., 0., 1.]]
    K = torch.FloatTensor(K).to(device)
    inv_K = torch.inverse(K).unsqueeze(0)
    K = K.unsqueeze(0)
    renderer = nr.Renderer(camera_mode='projection',
                            light_intensity_ambient=1.0,
                            light_intensity_directional=0.,
                            K=K, R=R, t=t,
                            near=renderer_min_depth, far=renderer_max_depth,
                            image_size=image_size, orig_size=image_size,
                            fill_back=fill_back,
                            background_color=[1.,1.,1.])
    return renderer


def get_grid(H, W, normalize=True):
    if normalize:
        h_range = torch.linspace(-1,1,H)
        w_range = torch.linspace(-1,1,W)
    else:
        h_range = torch.arange(0,H)
        w_range = torch.arange(0,W)
    grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).flip(2).float() # flip h,w to x,y
    return grid


def get_sphere_vtx(n_elev, n_azim):
    elevs = ((torch.arange(n_elev).view(n_elev, 1) +0.5) /n_elev *2 -1) *np.pi/2  # -pi/2~pi/2
    azims = ((torch.arange(n_azim).view(1, n_azim) +0.5) /n_azim *2 -1) *np.pi  # -pi~pi
    xs = elevs.cos() * azims.cos()
    ys = elevs.repeat(1, n_azim).sin()
    zs = elevs.cos() * azims.sin()
    vtx = torch.stack([xs, ys, zs], 2) # ExAx3
    return vtx


def get_sor_vtx(sor_curve, T):
    b, h, _ = sor_curve.shape
    rs, hs = sor_curve.unbind(2)  # BxH
    y = hs.view(b,h,1).repeat(1,1,T)  # BxHxT
    thetas = torch.linspace(-math.pi, math.pi, T+1)[:-1].to(sor_curve.device)  # T
    x = rs.unsqueeze(2) * thetas.cos().view(1,1,T)  # BxHxT
    z = rs.unsqueeze(2) * thetas.sin().view(1,1,T)  # BxHxT
    sor_vtx = torch.stack([x, y, z], 3)  # BxHxTx3
    return sor_vtx


def get_sor_full_face_idx(h, w):
    idx_map = torch.arange(h*w).reshape(h,w)  # HxW
    idx_map = torch.cat([idx_map, idx_map[:,:1]], 1)  # Hx(W+1), connect last column to first
    faces1 = torch.stack([idx_map[:h-1,:w], idx_map[1:,:w], idx_map[:h-1,1:w+1]], -1)  # (H-1)xWx3
    faces2 = torch.stack([idx_map[1:,1:w+1], idx_map[:h-1,1:w+1], idx_map[1:,:w]], -1)  # (H-1)xWx3
    return torch.stack([faces1, faces2], 0).int()  # 2x(H-1)xWx3


def get_sor_front_face_idx(h, w):
    sor_full_face_idx = get_sor_full_face_idx(h, w)  # 2x(H-1)x(W//2)x3
    return sor_full_face_idx[:,:,:w//2,:]


def get_sor_back_face_idx(h, w):
    sor_full_face_idx = get_sor_full_face_idx(h, w)  # 2x(H-1)x(W//2)x3
    return sor_full_face_idx[:,:,w//2:,:]


def get_tex_uv_grid(ts, h, w):
    uv_grid = get_grid(h, w, normalize=True)  # -1~1, HxWx(x,y)
    ab_grid = get_grid(ts, ts, normalize=False) / (ts-1)  # 0~1, txtx(x,y)
    ab_grid_uv_offsets = ab_grid * torch.FloatTensor([2/(w-1), 2/(h-1)]).view(1,1,2)

    tex_uv_grid1 = uv_grid[:-1,:-1,:].view(h-1, w-1, 1, 1, 2) + ab_grid_uv_offsets.view(1, 1, ts, ts, 2)  # (H-1)x(W-1)xtxtx2
    tex_uv_grid2 = uv_grid[1:,1:,:].view(h-1, w-1, 1, 1, 2) - ab_grid_uv_offsets.view(1, 1, ts, ts, 2)  # (H-1)x(W-1)xtxtx2
    tex_uv_grid = torch.stack([tex_uv_grid1, tex_uv_grid2], 0)  # 2x(H-1)x(W-1)xtxtx2
    return tex_uv_grid


def get_sor_vtx_normal(sor_vtx):
    sor_vtx = torch.nn.functional.pad(sor_vtx.permute(0,3,1,2), (1,1,0,0), mode='circular').permute(0,2,3,1)  # BxHx(T+2)x3

    tu = sor_vtx[:,1:-1,2:] - sor_vtx[:,1:-1,:-2]
    tv = sor_vtx[:,2:,1:-1] - sor_vtx[:,:-2,1:-1]
    normal = tu.cross(tv, dim=3)  # Bx(H-2)xTx3
    normal = torch.nn.functional.pad(normal.permute(0,3,1,2), (0,0,1,1), mode='replicate').permute(0,2,3,1)  # BxHxTx3
    normal = normal / (((normal**2).sum(3, keepdim=True))**0.5 + EPS)
    return normal  # BxHxTx3


def get_sor_quad_center_vtx(sor_vtx):
    ## shift to quad center for shading
    sor_vtx = torch.cat([sor_vtx, sor_vtx[:,:,:1]], 2)  # Hx(T+1), connect last column to first
    sor_quad_center_vtx = torch.nn.functional.avg_pool2d(sor_vtx.permute(0,3,1,2), kernel_size=2, stride=1, padding=0).permute(0,2,3,1)
    return sor_quad_center_vtx  # Bx(H-1)xTx3


def get_sor_quad_center_normal(sor_vtx):
    ## shift to quad center for shading
    sor_vtx = torch.cat([sor_vtx, sor_vtx[:,:,:1]], 2)  # Hx(T+1), connect last column to first

    tu = sor_vtx[:,:-1,1:] - sor_vtx[:,1:,:-1]
    tv = sor_vtx[:,1:,1:] - sor_vtx[:,:-1,:-1]
    normal = tu.cross(tv, dim=3)  # Bx(H-1)xTx3
    normal = normal / (((normal**2).sum(3, keepdim=True))**0.5 + EPS)
    return normal  # Bx(H-1)xTx3


def get_rotation_matrix(tx, ty, tz):
    m_x = torch.zeros((len(tx), 3, 3)).to(tx.device)
    m_y = torch.zeros((len(tx), 3, 3)).to(tx.device)
    m_z = torch.zeros((len(tx), 3, 3)).to(tx.device)

    m_x[:, 1, 1], m_x[:, 1, 2] = tx.cos(), -tx.sin()
    m_x[:, 2, 1], m_x[:, 2, 2] = tx.sin(), tx.cos()
    m_x[:, 0, 0] = 1

    m_y[:, 0, 0], m_y[:, 0, 2] = ty.cos(), ty.sin()
    m_y[:, 2, 0], m_y[:, 2, 2] = -ty.sin(), ty.cos()
    m_y[:, 1, 1] = 1

    m_z[:, 0, 0], m_z[:, 0, 1] = tz.cos(), -tz.sin()
    m_z[:, 1, 0], m_z[:, 1, 1] = tz.sin(), tz.cos()
    m_z[:, 2, 2] = 1
    return torch.matmul(m_z, torch.matmul(m_y, m_x))


def rotate_pts(pts, rotmat):
    return pts.matmul(rotmat.transpose(2,1))


def transform_sor(sor_vtx, rxyz=None, txy=None):
    if rxyz is not None:
        rx, ry, rz = rxyz.unbind(1)
        rotmat = get_rotation_matrix(rx, ry, rz).to(sor_vtx.device)
        sor_vtx = rotate_pts(sor_vtx, rotmat)  # BxNx3
    if txy is not None:
        tz = torch.zeros(len(txy), 1).to(txy.device)
        txyz = torch.cat([txy, tz], 1)
        sor_vtx = sor_vtx + txyz.unsqueeze(1)  # BxNx3
    return sor_vtx


def render_sor(renderer, sor_vtx, sor_faces, tex_im, tx_size=4, dim_inside=False):
    b, H, T, _ = sor_vtx.shape
    tex_uv_grid = get_tex_uv_grid(tx_size, H, T+1).to(sor_vtx.device)  # Bx2x(H-1)x(W-1)xtxtx2

    tx_cube = torch.nn.functional.grid_sample(tex_im, tex_uv_grid.view(1,-1,tx_size*tx_size,2).repeat(b,1,1,1), mode='bilinear', padding_mode="reflection", align_corners=False)  # Bx3xFxT^2
    tx_cube = tx_cube.permute(0,2,3,1).view(b,-1,1,tx_size,tx_size,3).repeat(1,1,tx_size,1,1,1)  # BxFxtxtxtx3

    sor_vtx = sor_vtx.view(b,-1,3)
    sor_faces = sor_faces.view(b,-1,3)
    if dim_inside:
        fill_back = renderer.fill_back
        renderer.fill_back = False
        sor_faces = torch.cat([sor_faces, sor_faces.flip(2)], 1)
        tx_cube = torch.cat([tx_cube, tx_cube*0.5], 1)
        im_rendered = renderer.render_rgb(sor_vtx, sor_faces, tx_cube)
        renderer.fill_back = fill_back
    else:
        im_rendered = renderer.render_rgb(sor_vtx, sor_faces, tx_cube)
    return im_rendered


def save_results(root, res, name=''):
    b = res['batch_size']
    keys = res.keys()
    os.makedirs(root, exist_ok=True)
    for i in range(b):
        idx = len(os.listdir(root)) +1
        fname = '%04d' %idx + name
        out_folder = os.path.join(root, fname)
        os.makedirs(out_folder)
        for key in keys:
            fpath = os.path.join(out_folder, fname+'_'+key)
            if key == 'batch_size':
                pass
            elif res[key].dim() == 4:
                im = np.uint8(res[key][i].permute(1,2,0).flip(2).numpy() *255)
                cv2.imwrite(fpath+'.png', im)
            else:
                np.savetxt(fpath+'.txt', res[key][i].numpy(), fmt='%.6f', delimiter=', ')


GAMMA = 2.2
def HDR2LDR(img):
    return img.clamp(min=EPS) **(1/GAMMA)


def LDR2HDR(img):
    return img.clamp(min=EPS) **GAMMA


def envmap_phong_shading(point3d, albedo, spec_albedo, normal, cam_loc, ambient, env_map, spec_alpha):
    b, c, tex_h, tex_w = albedo.shape
    _, h, w, _ = point3d.shape

    view_dirs = cam_loc.view(1,1,1,3) - point3d
    view_dirs = torch.nn.functional.normalize(view_dirs, p=2, dim=3, eps=EPS)

    _, n_elev, n_azim = env_map.shape
    l_dirs = get_sphere_vtx(n_elev, n_azim).unsqueeze(0).to(point3d.device)  # BxExAx3
    l_elevs = ((torch.arange(n_elev) +0.5) /n_elev *2 -1) *np.pi/2
    l_norm = l_elevs.cos().view(1, n_elev, 1).to(point3d.device) / n_elev / n_azim  # 1xEx1
    l_ints = env_map * l_norm *50

    cos_theta = ((-normal.unsqueeze(1)) * l_dirs.view(1,n_elev*n_azim,1,1,3)).sum(4, keepdim=True)
    diffuse = l_ints.view(b,n_elev*n_azim,1,1,-1) *cos_theta.clamp(0,1)  # BxLxHxWx1
    diffuse = torch.nn.functional.interpolate(diffuse.view(b*n_elev*n_azim,h,w,1).permute(0,3,1,2), (tex_h, tex_w), mode='bilinear', align_corners=False).view(b,n_elev*n_azim,1,tex_h,tex_w)

    reflect_dirs = -l_dirs.view(1,n_elev*n_azim,1,1,3) + 2*cos_theta*(-normal.unsqueeze(1))
    specular = (view_dirs.unsqueeze(1) * reflect_dirs).sum(4,keepdim=True).clamp(min=0) * (cos_theta>0)
    specular = (spec_alpha.view(b,1,1,1,-1)+1)/2/math.pi *l_ints.view(b,n_elev*n_azim,1,1,-1) * specular.clamp(min=EPS).pow(spec_alpha.view(b,1,1,1,-1))  # BxLxHxWx1
    specular = torch.nn.functional.interpolate(specular.view(b*n_elev*n_azim,h,w,1).permute(0,3,1,2), (tex_h, tex_w), mode='bilinear', align_corners=False).view(b,n_elev*n_azim,1,tex_h,tex_w)

    colors = (ambient.view(b,-1,1,1) + diffuse.sum(1)) *albedo + specular.sum(1) *spec_albedo
    return colors, diffuse, specular


def sg_to_env_map(sg_lights, n_elev=8, n_azim=16):
    b, n_sgls, _ = sg_lights.shape

    sgl_dirs = sg_lights[:,:,:3]
    sgl_lams = sg_lights[:,:,3:4]
    sgl_Fs = sg_lights[:,:,4:5]

    l_dirs = get_sphere_vtx(n_elev, n_azim).unsqueeze(0).to(sg_lights.device)  # BxExAx3
    exps = sgl_lams.view(b,n_sgls,1,1)* ((sgl_dirs.view(b,n_sgls,1,1,3)*l_dirs.view(1,1,n_elev,n_azim,3)).sum(4)-1)  # BxLxExA
    l_ints = sgl_Fs.view(b,n_sgls,1,1) * exps.exp()  # BxLxExA
    env_map = l_ints.sum(1) / n_sgls  # BxExA
    return env_map


def get_random_sor_curve(b, H):
    t1_bottom = torch.rand(b) *math.pi -math.pi  # -pi~0
    t1_top = torch.rand(b) *1.5*math.pi + 0.5*math.pi  # pi/2~pi*2
    amp1 = torch.rand(b) *0.3  # 0~0.3
    t2_bottom = torch.rand(b) *2*math.pi  # 0~pi*2
    t2_top = t2_bottom + torch.rand(b) *1.5*math.pi + 0.5*math.pi  # bottom + pi/2~pi*2
    amp2 = torch.rand(b) *0.1  # 0~0.1
    r0 = torch.rand(b) *0.2 + 0.1  # 0.02~0.3

    ts = torch.linspace(0,1,H)
    t1s = (1-ts.view(1,H)) * t1_bottom.view(b,1) +  ts.view(1,H) * t1_top.view(b,1)
    sin1 = amp1.view(b,1) * (t1s.sin() +1)
    t2s = (1-ts.view(1,H)) * t2_bottom.view(b,1) +  ts.view(1,H) * t2_top.view(b,1)
    sin2 = amp2.view(b,1) * (t2s.sin() +1)
    r_col = sin1 + sin2 + r0.view(b,1)  # 0.1~1
    r_col = r_col / r_col.max(1)[0].view(b,1) *0.75  # normalize to 0.75

    h_scale = torch.zeros(b) + 0.75
    h_col = torch.linspace(-1,1,H).view(1,H) *h_scale.view(b,1)

    sor_curve = torch.stack([r_col, h_col], 2)  # BxHx(r,h)
    return sor_curve


def get_random_pitch(b):
    return torch.rand(b) *20 /180*math.pi  # 0~20


def get_random_ambient(b):
    return torch.rand(b) *0.4 +0.1  # 0.1~0.5


def get_random_spec_albedo(b):
    return torch.rand(b) *0.9 +0.1  # 0.3~1


def get_random_spec_alpha(b):
    return (torch.rand(b) *13+1)**2  # 1~196


def get_random_sg_lights(b):
    n_sgls = 3
    sgl_dirs = torch.rand(b, n_sgls, 3) *2-1

    sgl_dirs[:,:,1] = -sgl_dirs[:,:,1].abs()  # upper only
    sgl_dirs[:,:,2] = -sgl_dirs[:,:,2].abs()  # front only

    sgl_dirs = torch.nn.functional.normalize(sgl_dirs, p=2, dim=2, eps=EPS)
    sgl_lams = torch.rand(b, n_sgls) *30+10
    sgl_Fs = (torch.rand(b, n_sgls) *0.3+0.1) *sgl_lams**0.5
    sg_lights = torch.cat([sgl_dirs, sgl_lams.unsqueeze(2), sgl_Fs.unsqueeze(2)], 2)
    return sg_lights


def get_random_env_map_ambient(b):
    return torch.rand(b) *0.03  # 0~0.1


def random_crop(im_pil, size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)):
    tfs_crop = torchvision.transforms.RandomResizedCrop(size, scale=scale, ratio=ratio)
    return tfs_crop(im_pil)


def random_color_jitter(im_pil, brightness=0, contrast=0, saturation=0, hue=0):
    tfs_jitter = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)
    return tfs_jitter(im_pil)


def generate(cc0_tex_dir, out_dir):
    H = 32
    T = 96
    image_size = 256
    fov = 10
    envmap_n_elev = 32
    envmap_n_azim = 96
    tex_im_h = 256
    tex_im_w = 768
    num_im_per_tex = 5

    device = 'cuda:0'
    b = 1
    oriz = 5
    cam_loc = torch.FloatTensor([0,0,-oriz]).to(device)
    lim = math.tan(fov/2/180*math.pi) *oriz
    max_depth = oriz + lim
    min_depth = oriz - lim
    renderer = get_renderer(world_ori=[0,0,oriz], image_size=image_size, fov=fov, fill_back=True)
    sor_faces = get_sor_full_face_idx(H, T).repeat(b,1,1,1,1).to(device)  # Bx2x(H-1)xWx3
    tx_size = 8
    tex_uv_grid = get_tex_uv_grid(tx_size, H, T+1).repeat(b,1,1,1,1,1,1).to(device)  # Bx2x(H-1)x(W-1)xtxtx2

    tex_im_list = sorted(glob(os.path.join(cc0_tex_dir, '**/*Color.jpg'), recursive=True))
    num_tex_im = len(tex_im_list)
    for i, tex_fpath in enumerate(tex_im_list):
        print(f'\n\nTex {i}/{num_tex_im}:')

        tex_im_pil = Image.open(tex_fpath).convert('RGB')

        for j in range(num_im_per_tex):
            tex_im_crop_pil = random_crop(tex_im_pil, size=(tex_im_h, tex_im_w//2), scale=(0.04, 1), ratio=(0.75, 1.3333333333333333))
            tex_im_crop_pil = random_color_jitter(tex_im_crop_pil, brightness=(1.0,1.5), contrast=(1.0,2.0), saturation=0., hue=0.5)
            tex_im_crop = torch.FloatTensor(np.array(tex_im_crop_pil) / 255).to(device)
            tex_im_crop = tex_im_crop.permute(2,0,1).unsqueeze(0)
            tex_im_crop = LDR2HDR(tex_im_crop)

            sor_curve = get_random_sor_curve(b,H).to(device) *lim  # BxHx2
            sor_vtx = get_sor_vtx(sor_curve, T)  # BxHxTx3
            pitch = get_random_pitch(b).to(device)
            rxyz = torch.stack([pitch, torch.zeros_like(pitch), torch.zeros_like(pitch)], 1)
            posed_sor_vtx = transform_sor(sor_vtx.view(b,-1,3), rxyz, txy=None).view(b,H,T,3)

            depth_rendered = renderer.render_depth(posed_sor_vtx.view(b,-1,3), sor_faces.view(b,-1,3)).clamp(min_depth, max_depth)
            mask_rendered = renderer.render_silhouettes(posed_sor_vtx.view(b,-1,3), sor_faces.view(b,-1,3))

            normal_map = get_sor_quad_center_normal(posed_sor_vtx)  # Bx(H-1)xTx3
            normal_tx_cube = torch.nn.functional.grid_sample(normal_map.permute(0,3,1,2), tex_uv_grid.view(b,-1,tx_size*tx_size,2), mode='bilinear', padding_mode="border", align_corners=False)  # Bx3xFxT^2
            normal_tx_cube = normal_tx_cube / (normal_tx_cube**2).sum(1,keepdim=True)**0.5 /2+0.5
            normal_tx_cube = normal_tx_cube.permute(0,2,3,1).view(b,-1,1,tx_size,tx_size,3).repeat(1,1,tx_size,1,1,1)  # BxFxtxtxtx3
            normal_rendered = renderer.render_rgb(posed_sor_vtx.view(b,-1,3), sor_faces.view(b,-1,3), normal_tx_cube).clamp(0, 1)

            ## sample lighting
            albedo = torch.cat([tex_im_crop, tex_im_crop.flip(3)], 3)
            ambient = get_random_ambient(b).to(device) *0
            spec_albedo = get_random_spec_albedo(b).to(device)
            spec_alpha = get_random_spec_alpha(b).to(device)
            sg_lights = get_random_sg_lights(b).to(device)

            posed_sor_vtx_map = get_sor_quad_center_vtx(posed_sor_vtx)  # Bx(H-1)xTx3
            env_map = sg_to_env_map(sg_lights, n_elev=envmap_n_elev, n_azim=envmap_n_azim)
            env_map_ambient = get_random_env_map_ambient(b).to(device)
            env_map = env_map + env_map_ambient.view(b,1,1)
            colors, diffuse, specular = envmap_phong_shading(posed_sor_vtx_map, albedo, spec_albedo.view(1,1,1,1), normal_map, cam_loc, ambient, env_map, spec_alpha)
            colors = HDR2LDR(colors)
            colors = colors.clamp(0,1)
            albedo = HDR2LDR(albedo)
            specular = specular * spec_albedo.view(1,1,1,1,1)

            im_rendered = render_sor(renderer, posed_sor_vtx, sor_faces, colors, tx_size=tx_size, dim_inside=True).clamp(0, 1)
            albedo_rendered = render_sor(renderer, posed_sor_vtx, sor_faces, albedo, tx_size=tx_size, dim_inside=True).clamp(0, 1)
            diffuse_map = diffuse.sum(1).clamp(0, 1).repeat(1,3,1,1)
            diffuse_rendered = render_sor(renderer, posed_sor_vtx, sor_faces, diffuse_map, tx_size=tx_size).clamp(0, 1)
            specular_map = specular.sum(1).clamp(0, 1).repeat(1,3,1,1)
            specular_rendered = render_sor(renderer, posed_sor_vtx, sor_faces, specular_map, tx_size=tx_size).clamp(0, 1)

            results = {}
            results['batch_size'] = b
            results['sor_curve'] = sor_curve.detach().cpu()
            results['pitch'] = pitch.detach().cpu().unsqueeze(1) /math.pi*180
            results['depth_rendered'] = (depth_rendered.detach().cpu().unsqueeze(1).repeat(1,3,1,1) - min_depth) / (max_depth-min_depth)
            results['mask_rendered'] = mask_rendered.detach().cpu().unsqueeze(1).repeat(1,3,1,1)

            results['im_rendered'] = im_rendered.detach().cpu()
            results['normal_map'] = normal_map.detach().cpu().permute(0,3,1,2) /2+0.5
            results['normal_rendered'] = normal_rendered.detach().cpu()
            results['albedo_map'] = albedo.detach().cpu()
            results['albedo_rendered'] = albedo_rendered.detach().cpu()
            results['texture_map'] = colors.detach().cpu()
            results['diffuse_map'] = diffuse_map.detach().cpu()
            results['diffuse_rendered'] = diffuse_rendered.detach().cpu()
            results['specular_map'] = specular_map.detach().cpu()
            results['specular_rendered'] = specular_rendered.detach().cpu()
            results['sg_lights'] = sg_lights.detach().cpu()
            results['ambient'] = ambient.detach().cpu().unsqueeze(1)
            results['spec_albedo'] = spec_albedo.detach().cpu().unsqueeze(1)
            results['spec_alpha'] = spec_alpha.detach().cpu().unsqueeze(1)
            results['env_map'] = env_map.clamp(0, 1).detach().cpu().repeat(1,3,1,1)
            results['env_map_ambient'] = env_map_ambient.detach().cpu().unsqueeze(1)

            tex_id = '_' + os.path.basename(os.path.dirname(tex_fpath)) + '_%02d' %j
            save_results(out_dir, results, name=tex_id)


if __name__ == '__main__':
    cc0_tex_dir = '../cc0_textures/PhotoTexturePBR'
    out_dir = '../syn_curv_sgl5_tex/rendering'

    for split in ['train', 'test']:
        print(f'Generating {split} set...')
        generate(os.path.join(cc0_tex_dir, split), os.path.join(out_dir, split))

    os.symlink('test', os.path.join(out_dir, 'val'))
