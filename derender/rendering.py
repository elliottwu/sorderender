import math
import torch
import neural_renderer as nr
from .utils import *


EPS = 1e-7


def get_rotation_matrix(rxyz):
    rx, ry, rz = rxyz.unbind(1)
    m_x, m_y, m_z = torch.zeros(3, len(rxyz), 3, 3).to(rxyz.device).unbind(0)

    m_x[:, 1, 1], m_x[:, 1, 2] = rx.cos(), -rx.sin()
    m_x[:, 2, 1], m_x[:, 2, 2] = rx.sin(), rx.cos()
    m_x[:, 0, 0] = 1

    m_y[:, 0, 0], m_y[:, 0, 2] = ry.cos(), ry.sin()
    m_y[:, 2, 0], m_y[:, 2, 2] = -ry.sin(), ry.cos()
    m_y[:, 1, 1] = 1

    m_z[:, 0, 0], m_z[:, 0, 1] = rz.cos(), -rz.sin()
    m_z[:, 1, 0], m_z[:, 1, 1] = rz.sin(), rz.cos()
    m_z[:, 2, 2] = 1
    return torch.matmul(m_z, torch.matmul(m_y, m_x))


def rotate_pts(pts, rotmat):
    return pts.matmul(rotmat.transpose(2,1))  # BxNx3


def translate_pts(pts, txyz):
    return pts + txyz.unsqueeze(1)  # BxNx3


def transform_pts(pts, rxyz=None, txy=None):
    original_shape = pts.shape
    pts = pts.view(original_shape[0],-1,3)
    if rxyz is not None:
        rotmat = get_rotation_matrix(rxyz).to(pts.device)
        pts = rotate_pts(pts, rotmat)  # BxNx3
    if txy is not None:
        tz = torch.zeros(len(txy), 1).to(txy.device)
        txyz = torch.cat([txy, tz], 1)
        pts = translate_pts(pts, txyz)  # BxNx3
    return pts.view(*original_shape)


def get_sphere_vtx(n_elev, n_azim):
    elevs = ((torch.arange(n_elev).view(n_elev, 1) +0.5) /n_elev *2 -1) *math.pi/2  # -pi/2~pi/2
    azims = ((torch.arange(n_azim).view(1, n_azim) +0.5) /n_azim *2 -1) *math.pi  # -pi~pi
    xs = elevs.cos() * azims.cos()
    ys = elevs.repeat(1, n_azim).sin()
    zs = elevs.cos() * azims.sin()
    vtx = torch.stack([xs, ys, zs], 2) # ExAx3
    return vtx


def get_sor_curve(r_col, h_scale):
    b, H = r_col.shape
    h_col = torch.linspace(-1,1,H).view(1,H).to(h_scale.device) *h_scale.view(b,1)
    sor_curve = torch.stack([r_col, h_col], 2)  # BxHx(r,h)
    return sor_curve


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


def vtx_3d_to_2d(vtx, K, world_ori, imH, imW):
    b, n, _ = vtx.shape
    vtx = vtx + torch.FloatTensor(world_ori).to(vtx.device).view(1,1,3)
    vtx_2d = vtx / vtx[...,2:]
    vtx_2d = vtx_2d.matmul(K.to(vtx.device).transpose(2,1))[:,:,:2]
    WH = torch.FloatTensor([imW, imH]).to(vtx.device).view(1,1,2)
    vtx_2d = (vtx_2d / WH) *2.-1.  # normalize to -1~1
    return vtx_2d


def unwrap_sor_front_tex_im(sor_vtx, tex_im_h, tex_im_w, K, world_ori, im):
    b, H, T, _ = sor_vtx.shape
    _, _, h, w = im.shape
    grid_xyz = torch.nn.functional.interpolate(sor_vtx.permute(0,3,1,2), (tex_im_h, tex_im_w), mode='bilinear', align_corners=False).permute(0,2,3,1)
    tex_uv_grid = vtx_3d_to_2d(grid_xyz.view(b,-1,3), K, world_ori, h, w).view(b,tex_im_h,tex_im_w,2)  # BxHxWx2
    front_tex_im = torch.nn.functional.grid_sample(im, tex_uv_grid, mode='bilinear')  # Bx3xHxW
    return front_tex_im


GAMMA = 2.2
def tonemap(img):
    return img.clamp(min=EPS) **(1/GAMMA)


def gamma(img):
    return img.clamp(min=EPS) **GAMMA


def compose_shading(diff_albedo, diff_map, spec_albedo, spec_map):
    b, c, tex_h, tex_w = diff_albedo.shape
    diff_map = torch.nn.functional.interpolate(diff_map, (tex_h, tex_w), mode='bilinear', align_corners=False)
    spec_map = torch.nn.functional.interpolate(spec_map, (tex_h, tex_w), mode='bilinear', align_corners=False)
    colors = diff_map * diff_albedo + spec_map * spec_albedo
    colors = tonemap(colors)
    return colors


def envmap_phong_shading(point3d, normal, cam_loc, env_map, spec_alpha):
    b, h, w, _ = point3d.shape

    view_dirs = cam_loc.view(1,1,1,3) - point3d
    view_dirs = torch.nn.functional.normalize(view_dirs, p=2, dim=3, eps=EPS)

    _, n_elev, n_azim = env_map.shape
    l_dirs = get_sphere_vtx(n_elev, n_azim).unsqueeze(0).to(point3d.device)  # BxExAx3
    l_elevs = ((torch.arange(n_elev) +0.5) /n_elev *2 -1) *math.pi/2
    l_norm = l_elevs.cos().view(1, n_elev, 1).to(point3d.device) / n_elev / n_azim  # 1xEx1
    l_ints = env_map * l_norm *50

    cos_theta = ((-normal.unsqueeze(1)) * l_dirs.view(1,n_elev*n_azim,1,1,3)).sum(4, keepdim=True)
    diffuse = l_ints.view(b,n_elev*n_azim,1,1,-1) *cos_theta.clamp(0,1)  # BxLxHxWx1
    diffuse = diffuse.sum(1).permute(0,3,1,2)

    reflect_dirs = -l_dirs.view(1,n_elev*n_azim,1,1,3) + 2*cos_theta*(-normal.unsqueeze(1))
    specular = (view_dirs.unsqueeze(1) * reflect_dirs).sum(4,keepdim=True).clamp(min=0) * (cos_theta>0)
    specular = (spec_alpha.view(b,1,1,1,-1)+1)/2/math.pi *l_ints.view(b,n_elev*n_azim,1,1,-1) * specular.clamp(min=EPS).pow(spec_alpha.view(b,1,1,1,-1))  # BxLxHxWx1
    specular = specular.sum(1).permute(0,3,1,2)

    return diffuse, specular


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
    fx = image_size /2 /(math.tan(fov/2 *math.pi/180))
    fy = image_size /2 /(math.tan(fov/2 *math.pi/180))
    cx = image_size /2
    cy = image_size /2
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
                            # background_color=[0.5,0.5,0.5])
                            background_color=[1.,1.,1.])
    return renderer


def render_sor(renderer, sor_vtx, sor_faces, tex_im, tx_size=4, dim_inside=False, render_normal=False):
    # b, H, T, _ = sor_vtx.shape
    b, _, H_, T_, _ = sor_faces.shape
    tex_uv_grid = get_tex_uv_grid(tx_size, H_+1, T_+1).to(sor_vtx.device)  # Bx2xHxWxtxtx2

    if render_normal:
        tx_cube = torch.nn.functional.grid_sample(tex_im, tex_uv_grid.view(1,-1,tx_size*tx_size,2).repeat(b,1,1,1), mode='bilinear', padding_mode="border", align_corners=False)  # Bx3xFxT^2
        tx_cube = tx_cube / (tx_cube**2).sum(1,keepdim=True)**0.5 /2+0.5
    else:
        tx_cube = torch.nn.functional.grid_sample(tex_im, tex_uv_grid.view(1,-1,tx_size*tx_size,2).repeat(b,1,1,1), mode='bilinear', padding_mode="reflection", align_corners=False)  # Bx3xFxT^2
    tx_cube = tx_cube.permute(0,2,3,1).view(b,-1,1,tx_size,tx_size,3).repeat(1,1,tx_size,1,1,1)  # BxFxtxtxtx3

    sor_vtx = sor_vtx.reshape(b,-1,3)
    sor_faces = sor_faces.reshape(b,-1,3)
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


def render_material(renderer, spec_alpha, spec_albedo_scalar):
    b = spec_alpha.size(0)
    device = spec_alpha.device
    H = 64
    T = 128
    tex_im_h = 256
    tex_im_w = 512
    tx_size = 8
    env_map = torch.ones(b,16,48).to(device) *0.02
    env_map[:, 3, 5] = 4
    albedo = torch.ones(b, 3, tex_im_h, tex_im_w).to(device) *1
    ori_z = 5
    cam_loc = torch.FloatTensor([0,0,-ori_z]).to(device)
    fov = 10
    max_range = math.tan(fov/2/180*math.pi) *ori_z

    sor_faces = get_sor_full_face_idx(H, T).repeat(b,1,1,1,1).to(device)  # Bx2x(H-1)xWx3
    tex_uv_grid = get_tex_uv_grid(tx_size, H, T+1).repeat(b,1,1,1,1,1,1).to(device)  # Bx2x(H-1)x(W-1)xtxtx2
    sor_vtx = get_sphere_vtx(H, T).repeat(b,1,1,1).to(device) *max_range *0.8
    sor_vtx_map = get_sor_quad_center_vtx(sor_vtx)
    normal_map = get_sor_quad_center_normal(sor_vtx)  # Bx(H-1)xTx3

    diffuse, specular = envmap_phong_shading(sor_vtx_map, normal_map, cam_loc, env_map, spec_alpha)
    diffuse = torch.nn.functional.interpolate(diffuse, (tex_im_h, tex_im_w), mode='bilinear', align_corners=False)
    specular = torch.nn.functional.interpolate(specular, (tex_im_h, tex_im_w), mode='bilinear', align_corners=False)
    tex_im = diffuse * albedo + specular * spec_albedo_scalar.view(b,1,1,1)
    im_rendered = render_sor(renderer, sor_vtx, sor_faces, tex_im, tx_size=tx_size, dim_inside=False).clamp(0,1)
    return im_rendered


def render_novel_view(renderer, canon_sor_vtx, sor_faces, albedo, spec_albedo, spec_alpha, env_map):
    b = canon_sor_vtx.size(0)
    device = canon_sor_vtx.device
    tx_size = 8
    ori_z = 5
    cam_loc = torch.FloatTensor([0,0,-ori_z]).to(device)

    rxyz = torch.FloatTensor([0, 30, 0]).repeat(b,1).to(device) /180*math.pi
    sor_vtx = transform_pts(canon_sor_vtx, rxyz, None)
    rxyz = torch.FloatTensor([30, 0, 0]).repeat(b,1).to(device) /180*math.pi
    sor_vtx = transform_pts(sor_vtx, rxyz, None)

    sor_vtx_map = get_sor_quad_center_vtx(sor_vtx)
    normal_map = get_sor_quad_center_normal(sor_vtx)  # Bx(H-1)xTx3
    diffuse, specular = envmap_phong_shading(sor_vtx_map, normal_map, cam_loc, env_map, spec_alpha)

    tex_im = compose_shading(albedo, diffuse, spec_albedo, specular).clamp(0,1)
    im_rendered = render_sor(renderer, sor_vtx, sor_faces, tex_im, tx_size=tx_size, dim_inside=True).clamp(0, 1)
    mask_rendered = renderer.render_silhouettes(sor_vtx.view(b,-1,3), sor_faces.view(b,-1,3))
    return im_rendered, mask_rendered
