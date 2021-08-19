import os
import glob
import random
import numpy as np
import cv2
import torch
import torchvision
import imageio
import yaml
import zipfile


def setup_runtime(args):
    """Load configs, initialize CUDA, CuDNN and the random seeds."""

    # Setup CUDA
    cuda_device_id = args.gpu
    if cuda_device_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device_id
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # Setup random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ## Load config
    cfgs = {}
    if args.config is not None and os.path.isfile(args.config):
        cfgs = load_yaml(args.config)

    cfgs['config'] = args.config
    cfgs['seed'] = args.seed
    cfgs['num_workers'] = args.num_workers
    cfgs['device'] = 'cuda:0' if torch.cuda.is_available() and cuda_device_id is not None else 'cpu'

    print(f"Environment: GPU {cuda_device_id} seed {args.seed} number of workers {args.num_workers}")
    return cfgs


def load_yaml(path):
    print(f"Loading configs from {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def dump_yaml(path, cfgs):
    print(f"Saving configs to {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        return yaml.safe_dump(cfgs, f)


def clean_checkpoint(checkpoint_dir, keep_num=2):
    if keep_num > 0:
        names = list(sorted(
            glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.pth'))
        ))
        if len(names) > keep_num:
            for name in names[:-keep_num]:
                print(f"Deleting obslete checkpoint file {name}")
                os.remove(name)


def archive_code(arc_path, file_patterns=['*.py']):
    print(f"Archiving code to {arc_path}")
    os.makedirs(os.path.dirname(arc_path), exist_ok=True)
    zipf = zipfile.ZipFile(arc_path, 'w', zipfile.ZIP_DEFLATED)
    cur_dir = os.getcwd()
    flist = []
    for ftype in file_patterns:
        flist.extend(glob.glob(os.path.join(cur_dir, '**', '*'+ftype), recursive=True))
    [zipf.write(f, arcname=f.replace(cur_dir,'archived_code', 1)) for f in flist]
    zipf.close()


def get_model_device(model):
    return next(model.parameters()).device


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def save_results(root, res, name=''):
    b = res['batch_size']
    keys = res.keys()
    os.makedirs(root, exist_ok=True)
    for i in range(b):
        folders = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))]
        idx = len(folders) +1
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


def save_images(out_fold, imgs, prefix='', suffix='', sep_folder=True, ext='.png'):
    if sep_folder:
        out_fold = os.path.join(out_fold, suffix)
    os.makedirs(out_fold, exist_ok=True)
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''
    offset = len(glob.glob(os.path.join(out_fold, prefix+'*'+suffix+ext))) +1

    imgs = imgs.transpose(0,2,3,1)
    for i, img in enumerate(imgs):
        if 'depth' in suffix:
            im_out = np.uint16(img[...,::-1]*65535.)
        else:
            im_out = np.uint8(img[...,::-1]*255.)
        cv2.imwrite(os.path.join(out_fold, prefix+'%05d'%(i+offset)+suffix+ext), im_out)


def save_videos(out_fold, imgs, prefix='', suffix='', sep_folder=True, ext='.mp4', cycle=False, fps=5):
    if sep_folder:
        out_fold = os.path.join(out_fold, suffix)
    os.makedirs(out_fold, exist_ok=True)
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''
    offset = len(glob.glob(os.path.join(out_fold, prefix+'*'+suffix+ext))) +1

    imgs = imgs.transpose(0,1,3,4,2)  # BxTxCxHxW -> BxTxHxWxC
    for i, fs in enumerate(imgs):
        if cycle:
            fs = np.concatenate([fs, fs[::-1]], 0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'avc1')
        vid = cv2.VideoWriter(os.path.join(out_fold, prefix+'%05d'%(i+offset)+suffix+ext), fourcc, fps, (fs.shape[2], fs.shape[1]))
        [vid.write(np.uint8(f[...,::-1]*255.)) for f in fs]
        vid.release()


def save_gifs(out_fold, imgs, prefix='', suffix='', sep_folder=True, ext='.gif', cycle=False):
    if sep_folder:
        out_fold = os.path.join(out_fold, suffix)
    os.makedirs(out_fold, exist_ok=True)
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''
    offset = len(glob.glob(os.path.join(out_fold, prefix+'*'+suffix+ext))) +1

    imgs = imgs.transpose(0,1,3,4,2)  # BxTxCxHxW -> BxTxHxWxC
    for i, fs in enumerate(imgs):
        if cycle:
            fs = np.concatenate([fs, fs[::-1]], 0)
        fs = np.uint8(fs*255.)  # TxHxWxC
        imageio.mimsave(os.path.join(out_fold, prefix+'%05d'%(i+offset)+suffix+ext), fs, fps=5)


def save_txt(out_fold, data, prefix='', suffix='', sep_folder=True, ext='.txt'):
    if sep_folder:
        out_fold = os.path.join(out_fold, suffix)
    os.makedirs(out_fold, exist_ok=True)
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''
    offset = len(glob.glob(os.path.join(out_fold, prefix+'*'+suffix+ext))) +1

    [np.savetxt(os.path.join(out_fold, prefix+'%05d'%(i+offset)+suffix+ext), d, fmt='%.6f', delimiter=', ') for i,d in enumerate(data)]


def compute_sc_inv_err(d_pred, d_gt, mask=None):
    b = d_pred.size(0)
    diff = d_pred - d_gt
    if mask is not None:
        diff = diff * mask
        avg = diff.view(b, -1).sum(1) / (mask.view(b, -1).sum(1))
        score = (diff - avg.view(b,1,1))**2 * mask
    else:
        avg = diff.view(b, -1).mean(1)
        score = (diff - avg.view(b,1,1))**2
    return score  # masked error maps


def compute_angular_distance(n1, n2, mask=None):
    dist = (n1*n2).sum(3).clamp(-1,1).acos() /np.pi*180
    return dist*mask if mask is not None else dist


def save_scores(out_path, scores, header=''):
    print('Saving scores to %s' %out_path)
    np.savetxt(out_path, scores, fmt='%.8f', delimiter=',\t', header=header)


def get_grid(H, W, normalize=True):
    if normalize:
        h_range = torch.linspace(-1,1,H)
        w_range = torch.linspace(-1,1,W)
    else:
        h_range = torch.arange(0,H)
        w_range = torch.arange(0,W)
    grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).flip(2).float() # flip h,w to x,y
    return grid


def get_patches(im, num_patch=8, patch_size=64, scale=(0.25,0.5)):
    b, c, h, w = im.shape
    wh = torch.rand(b*num_patch, 2) *(scale[1]-scale[0]) + scale[0]
    xy0 = torch.rand(b*num_patch, 2) *(1-wh) *2 -1  # -1~1-wh
    xy_grid = get_grid(patch_size, patch_size, normalize=True).repeat(b*num_patch,1,1,1)  # -1~1
    xy_grid = xy0.view(b*num_patch,1,1,2) + (xy_grid+1) *wh.view(b*num_patch,1,1,2)
    patches = torch.nn.functional.grid_sample(im.repeat(num_patch,1,1,1), xy_grid.to(im.device), mode='bilinear', align_corners=False).view(num_patch,b,c,patch_size,patch_size).transpose(1,0)
    return patches  # BxNxCxHxW
