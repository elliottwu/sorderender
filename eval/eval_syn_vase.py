import os
import cv2
from glob import glob
import numpy as np
import torch


EPS = 1e-7


def load_imgs(flist):
    return torch.stack([torch.FloatTensor(cv2.imread(f) /255.).flip(2) for f in flist], 0).permute(0,3,1,2)


def load_txts(flist):
    return torch.stack([torch.FloatTensor(np.loadtxt(f, delimiter=',')) for f in flist], 0)


GAMMA = 2.2
def gamma(img):
    return img.clamp(min=EPS) **GAMMA


def compute_shift_inv_err(pred, gt, mask=None):
    b, c, h, w = pred.shape
    diff = pred - gt
    if mask is not None:
        diff = diff * mask
        avg = diff.reshape(b, -1).sum(1) / (mask.reshape(b, -1).sum(1))
        score = (diff - avg.view(b,1,1,1))**2
        # avg = diff.reshape(b, c, -1).sum(2) / (mask.reshape(b, c, -1).sum(2))
        # score = (diff - avg.view(b,c,1,1))**2
        score = (score * mask).reshape(b, -1).sum(1) / mask.reshape(b, -1).sum(1)
    else:
        avg = diff.view(b, -1).mean(1)
        score = (diff - avg.view(b,1,1,1))**2
        score = score.view(b, -1).mean(1)
    return score


def compute_scale_inv_err(pred, gt, mask=None):
    ## inputs are 0+
    b, c, h, w = pred.shape
    if mask is not None:
        numer = (pred * gt * mask).reshape(b, -1).sum(1)
        denom = (pred * pred * mask).reshape(b, -1).sum(1)
        scale = numer / denom
    else:
        numer = (pred * gt).reshape(b, -1).sum(1)
        denom = (pred * pred).reshape(b, -1).sum(1)
        scale = numer / denom
    score = compute_mse(pred * scale.view(b,1,1,1), gt, mask)
    return score


def compute_mse(pred, gt, mask=None):
    b = pred.size(0)
    score = (pred - gt)**2
    if mask is not None:
        score = (score * mask).reshape(b, -1).sum(1) / mask.reshape(b, -1).sum(1)
    else:
        score = score.view(b, -1).mean(1)
    return score


def compute_rmse(pred, gt, mask=None):
    return compute_mse(pred, gt, mask)**0.5


def compute_angular_distance(n1, n2, mask=None):
    b = n1.size(0)
    dist = (n1*n2).sum(3).clamp(-1,1).acos() /np.pi*180
    if mask is not None:
        dist = (dist * mask).reshape(b, -1).sum(1) / mask.reshape(b, -1).sum(1)
    else:
        dist = dist.reshape(b, -1).mean(1)
    return dist


def main(gt_dir, res_dir):
    ## load data
    print("Loading GT from %s" %gt_dir)
    gt_albedo_rendered = load_imgs(sorted(glob(os.path.join(gt_dir, '**/*_albedo_rendered.png'), recursive=True)))
    gt_masks = load_imgs(sorted(glob(os.path.join(gt_dir, '**/*_mask_rendered.png'), recursive=True)))
    gt_pitch = load_txts(sorted(glob(os.path.join(gt_dir, '**/*_pitch.txt'), recursive=True)))
    gt_normal_rendered = load_imgs(sorted(glob(os.path.join(gt_dir, '**/*_normal_rendered.png'), recursive=True)))
    gt_spec_alpha = load_txts(sorted(glob(os.path.join(gt_dir, '**/*_spec_alpha.txt'), recursive=True)))
    gt_spec_albedo = load_txts(sorted(glob(os.path.join(gt_dir, '**/*_spec_albedo.txt'), recursive=True)))
    gt_env_map = load_imgs(sorted(glob(os.path.join(gt_dir, '**/*_env_map.png'), recursive=True)))[:,:1]

    print("Loading preditions from %s" %res_dir)
    pred_albedo_rendered = load_imgs(sorted(glob(os.path.join(res_dir, 'albedo_rendered/*_albedo_rendered.png'), recursive=True)))
    pred_front_masks = load_imgs(sorted(glob(os.path.join(res_dir, 'front_mask_rendered/*_front_mask_rendered.png'), recursive=True)))
    pred_pose = load_txts(sorted(glob(os.path.join(res_dir, 'pose/*_pose.txt'), recursive=True)))
    pred_normal_rendered = load_imgs(sorted(glob(os.path.join(res_dir, 'normal_rendered/*_normal_rendered.png'), recursive=True)))
    pred_material = load_txts(sorted(glob(os.path.join(res_dir, 'material/*_material.txt'), recursive=True)))
    pred_env_map = load_imgs(sorted(glob(os.path.join(res_dir, 'env_map/*_env_map.png'), recursive=True)))[:,:1]

    ## sc. inv. albedo err, pose err, normal err, spec_alpha, spec_albedo, sc. inv. front env map
    print("Computing scores...")
    scores = {}
    both_masks = gt_masks * pred_front_masks
    scores['albedo_err_sie'] = compute_scale_inv_err(pred_albedo_rendered, gt_albedo_rendered, both_masks)
    scores['pose_err_rmse'] = compute_rmse(gt_pitch, pred_pose[:,0])
    scores['normal_err_ad'] = compute_angular_distance(gt_normal_rendered.permute(0,2,3,1) *2-1, pred_normal_rendered.permute(0,2,3,1) *2-1, both_masks[:,0])
    scores['spec_alpha_rmse'] = compute_rmse(gt_spec_alpha, pred_material[:,0])
    scores['spec_albedo_rmse'] = compute_rmse(gt_spec_albedo, pred_material[:,1])
    _, _, eh, ew = gt_env_map.shape
    pred_env_map = torch.nn.functional.interpolate(pred_env_map, (eh, ew), mode='bilinear', align_corners=False)
    scores['env_map_err_sie'] = compute_scale_inv_err(pred_env_map[:,:,:,:ew//2], gt_env_map[:,:,:,:ew//2])

    out_fold = os.path.join(res_dir, 'scores')
    os.makedirs(out_fold, exist_ok=True)
    for i in range(len(scores['albedo_err_sie'])):
        score = [[k+ ': %.6f'%v[i]] for k,v in scores.items()]
        np.savetxt(os.path.join(out_fold, '%05d_scores.txt'%(i+1)), score, fmt='%s', delimiter='')
    score = [[k+ ': %.6f'%v.mean() + ' +- %.6f'%v.std()] for k,v in scores.items()]
    np.savetxt(os.path.join(res_dir, 'all_scores.txt'), score, fmt='%s', delimiter='')


if __name__ == '__main__':
    gt_dir = 'data/syn_vases/rendering/test'
    res_dir = 'results/syn_vase_pretrained/test_results_checkpoint200'
    main(gt_dir, res_dir)
