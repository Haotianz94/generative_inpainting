import torch
import torch.nn as nn
import torchvision
import sys

import numpy as np
from PIL import Image
import PIL
import numpy as np
import cv2
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim


def get_model_num_parameters(model):
    total_num=0
    if type(model) == type(dict()):
        for key in model:
            for p in model[key].parameters():
                total_num+=p.nelement()
    else:
        for p in model.parameters():
            total_num+=p.nelement()
    return total_num


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False


def get_noise(batch_size, input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (`batch_size` x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [batch_size, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid':
        assert batch_size == 1
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input


def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np): 
    '''Converts image in np.array format to PIL image.
    
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    if img_np.shape[0] == 1:
        img_np = img_np[0]
    elif img_np.shape[0] == 3:
        img_np = img_np.transpose(1, 2, 0)

    if img_np.dtype == np.float32:
        img_np = np.clip(img_np*255,0,255).astype(np.uint8)

    return Image.fromarray(img_np)


def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From B x C x W x H [0..1] to  B x C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)


def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From B x C x W x H [0..1] to  B x C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()


def upsample(img_np, resize):
    img_np_resize = []
    h, w = img_np.shape[2:]
    img_np = img_np.transpose(0, 2, 3, 1)
    h = int(h * resize)
    w = int(w * resize)
    for img in img_np:
        img_resize = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        img_np_resize.append(img_resize)
    return np.array(img_np_resize).transpose(0, 3, 1, 2)


def warp_torch(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.ones(x.size())
    if x.is_cuda:
        mask = mask.cuda()
    mask = nn.functional.grid_sample(mask, vgrid)
    
    mask[mask<0.999] = 0
    mask[mask>0] = 1
    
    return output, mask

                                         
def warp_np(x, flo):
    if x.ndim != 4:
        assert(x.ndim == 3)
        # add one dimention for single image
        x = x[None, ...]
        flo = flo[None, ...]
        add_dim = True
    else:
        add_dim = False

    output, mask = warp_torch(np_to_torch(x), np_to_torch(flo))
    if add_dim:
        return output.numpy()[0], mask.numpy()[0]
    else:
        return output.numpy(), mask.numpy()


def check_flow_occlusion(flow_f, flow_b):
    def get_occlusion(flow1, flow2):
        grid_flow = grid + flow1
        grid_flow[0,:,:] = 2.0*grid_flow[0,:,:]/max(W-1,1)-1.0
        grid_flow[1,:,:] = 2.0*grid_flow[1,:,:]/max(H-1,1)-1.0
        grid_flow = grid_flow.permute(1,2,0)        
        flow2_inter = nn.functional.grid_sample(flow2[None, ...], grid_flow[None, ...])[0]
        score = torch.exp(- torch.sum((flow1 + flow2_inter) ** 2, dim=0) / 2.)
        occlusion = (score > 0.5)
        return occlusion[None, ...].float()

    C, H, W = flow_f.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,H,W)
    yy = yy.view(1,H,W)
    grid = torch.cat((xx,yy),0).float()

    occlusion_f = get_occlusion(flow_f, flow_b)
    occlusion_b = get_occlusion(flow_b, flow_f)
    flow_f = torch.cat((flow_f, occlusion_f), 0)
    flow_b = torch.cat((flow_b, occlusion_b), 0)

    return flow_f, flow_b


def measure_psnr_ssim(img_batch, mask_batch, out_batch, compare_gt):
    psnr = np.zeros(3)
    ssim = np.zeros(3)
    batch_size = img_batch.shape[0]
    img_batch = img_batch.transpose(0, 2, 3, 1)
    mask_batch = mask_batch.transpose(0, 2, 3, 1)
    out_batch = out_batch.transpose(0, 2, 3, 1)
    # img_nonhole_batch = (img_batch * (1 - mask_batch))
    # out_nonhole_batch = (out_batch * (1 - mask_batch))
    # if compare_gt:
    #     img_hole_batch = (img_batch * mask_batch)
    #     out_hole_batch = (out_batch * mask_batch)
    # for i in range(batch_size):
    #     num_hole = np.count_nonzero(mask_batch[i])
    #     num_nonhole = np.count_nonzero(1 - mask_batch[i])

    #     psnr_nonhole = compute_psnr(img_nonhole_batch[i], out_nonhole_batch[i], num_nonhole * 3)
    #     ssim_nonhole = compute_ssim(img_nonhole_batch[i], out_nonhole_batch[i], 1. * num_hole / num_nonhole)
    #     psnr[0] += psnr_nonhole
    #     ssim[0] += ssim_nonhole
    #     if compare_gt:
    #         if num_hole == 0:
    #             psnr[1] += psnr_nonhole
    #             ssim[1] += ssim_nonhole
    #             psnr[2] += psnr_nonhole
    #             ssim[2] += ssim_nonhole
    #         else:    
    #             psnr[1] += compute_psnr(img_hole_batch[i], out_hole_batch[i], num_hole * 3)
    #             ssim[1] += compute_ssim(img_hole_batch[i], out_hole_batch[i], 1. * num_nonhole / num_hole)
    #             psnr[2] += compute_psnr(img_batch[i], out_batch[i], (num_hole + num_nonhole) * 3)
    #             ssim[2] += compute_ssim(img_batch[i], out_batch[i], 0)

    # psnr /= batch_size
    # ssim /= batch_size

    # only compute on middle frame
    idx = batch_size // 2
    img_nonhole = (img_batch[idx] * (1 - mask_batch[idx]))
    out_nonhole = (out_batch[idx] * (1 - mask_batch[idx]))
    if compare_gt:
        img_hole = (img_batch[idx] * mask_batch[idx])
        out_hole = (out_batch[idx] * mask_batch[idx])
    num_hole = np.count_nonzero(mask_batch[idx])
    num_nonhole = np.count_nonzero(1 - mask_batch[idx])

    psnr_nonhole = compute_psnr(img_nonhole, out_nonhole, num_nonhole * 3)
    ssim_nonhole = compute_ssim(img_nonhole, out_nonhole, 1. * num_hole / num_nonhole)
    psnr[0] = psnr_nonhole
    ssim[0] = ssim_nonhole
    if compare_gt:
        if num_hole == 0:
            psnr[1] = psnr_nonhole
            ssim[1] = ssim_nonhole
            psnr[2] = psnr_nonhole
            ssim[2] = ssim_nonhole
        else:    
            psnr[1] = compute_psnr(img_hole, out_hole, num_hole * 3)
            ssim[1] = compute_ssim(img_hole, out_hole, 1. * num_nonhole / num_hole)
            psnr[2] = compute_psnr(img_batch[idx], out_batch[idx], (num_hole + num_nonhole) * 3)
            ssim[2] = compute_ssim(img_batch[idx], out_batch[idx], 0)
    return psnr, ssim


def compute_psnr(im1, im2, num_pixel):
    mse = np.sum(np.square(im1 - im2)) / num_pixel
    return 10 * np.log10(1. / mse)


def compute_ssim(im1, im2, ratio):
    ssim = compare_ssim(im1, im2, multichannel=True)
    return ssim * (1 + ratio) - ratio


def measure_cpsnr(out_batch, mask_batch, flow_f1_batch, compare_gt):
    err = np.zeros(3)
    batch_size = out_batch.shape[0]
    if batch_size == 1:
        return err

    # warped_img, flowmask = warp_np(out_batch[1:], flow_f1_batch)
    # warped_mask, _ = warp_np(mask_batch[1:], flow_f1_batch)
    # warped_mask = (warped_mask > 0).astype(np.float32)
    # mask_union_inv = (1. - mask_batch[:-1]) * (1. - warped_mask) * flowmask
    # mask_union = (1 - (1. - mask_batch[:-1]) * (1. - warped_mask)) * flowmask
    # out_batch = (out_batch[:-1] * flowmask).transpose(0, 2, 3, 1)
    # warped_img = (warped_img * flowmask).transpose(0, 2, 3, 1)
    # mask_union_inv = mask_union_inv.transpose(0, 2, 3, 1)
    # mask_union = mask_union.transpose(0, 2, 3, 1)
    
    # out_nonhole_batch = out_batch * mask_union_inv
    # warped_nonhole_batch = warped_img * mask_union_inv
    # if compare_gt:
    #     out_hole_batch = out_batch * mask_union
    #     warped_hole_batch = warped_img * mask_union
    # for i in range(batch_size - 1):
    #     num_nonhole = np.count_nonzero(mask_union_inv[i])
    #     num_hole = np.count_nonzero(mask_union[i])

    #     psnr_nonhole = compute_psnr(out_nonhole_batch[i], warped_nonhole_batch[i], num_nonhole * 3)
    #     err[0] += psnr_nonhole
    #     if compare_gt:
    #         if num_hole == 0:
    #             err[1] += psnr_nonhole
    #             err[2] += psnr_nonhole
    #         else:    
    #             err[1] += compute_psnr(out_hole_batch[i], warped_hole_batch[i], num_hole * 3)
    #             err[2] += compute_psnr(out_batch[i], warped_img[i], (num_hole + num_nonhole) * 3)
    # err /= (batch_size - 1)

    # only compute on middle frame
    idx = batch_size // 2
    warped_img, flowmask = warp_np(out_batch[idx+1], flow_f1_batch[idx, :2, ...])
    warped_mask, _ = warp_np(mask_batch[idx+1], flow_f1_batch[idx, :2, ...])
    warped_mask = (warped_mask > 0).astype(np.float32)
    mask_union_inv = (1. - mask_batch[idx]) * (1. - warped_mask) * flow_f1_batch[idx, 2:3, ...] * flowmask
    mask_union = (1 - (1. - mask_batch[idx]) * (1. - warped_mask)) * flow_f1_batch[idx, 2:3, ...] * flowmask
    out_img = (out_batch[idx] * flowmask).transpose(1, 2, 0)
    warped_img = (warped_img * flowmask).transpose(1, 2, 0)
    mask_union_inv = mask_union_inv.transpose(1, 2, 0)
    mask_union = mask_union.transpose(1, 2, 0)
    num_nonhole = np.count_nonzero(mask_union_inv)
    num_hole = np.count_nonzero(mask_union)

    psnr_nonhole = compute_psnr(out_img * mask_union_inv, warped_img * mask_union_inv, num_nonhole * 3)
    err[0] = psnr_nonhole
    if compare_gt:
        if num_hole == 0:
            err[1] = psnr_nonhole
            err[2] = psnr_nonhole
        else:    
            err[1] = compute_psnr(out_img * mask_union, warped_img * mask_union, num_hole * 3)
            err[2] = compute_psnr(out_img, warped_img, (num_hole + num_nonhole) * 3)

    return err           


def flow_to_image(flow, mode='RGB'):
    """
    this function visualize the input flow
    :param flow: input flow in array
    :param mode: choose which color mode to visualize the flow (Y: Ycbcr, RGB: RGB color)
    :return: None
    """
    if mode == 'YCbCr':
        # Ccbcr color wheel
        u = flow[0, :, :]
        v = flow[1, :, :]

        maxu = -999.
        maxv = -999.
        minu = 999.
        minv = 999.

        UNKNOWN_FLOW_THRESH = 1e7
        idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
        u[idxUnknow] = 0
        v[idxUnknow] = 0

        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))

        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))

        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(-1, np.max(rad))

        u = u/(maxrad + np.finfo(float).eps)
        v = v/(maxrad + np.finfo(float).eps)

        img = compute_color(u, v)
        idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
        img[idx] = 0
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_HSV2RGB)
        return img 

    elif mode == 'HSV':
        h, w = flow.shape[1:]
        hsv = np.zeros((h, w, 3), dtype=np.float32)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[0, ...], flow[1, ...])
        hsv[..., 0] = ang / (2*np.pi) * 255
        # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = mag * 10 + 100
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img 

    elif mode == 'HEAT':
        mag, ang = cv2.cartToPolar(flow[0, ...], flow[1, ...])
        img = mag * 10 + 100
        return np.uint8(img) 

    elif mode == 'UVmap':
        h, w = flow.shape[1:]
        umap = np.zeros((h, w, 3), dtype=np.float32)
        vmap = np.zeros((h, w, 3), dtype=np.float32)
        umap[:, :, 2] = (flow[0] < 0) * np.fabs(flow[0]) * 10
        umap[:, :, 0] = (flow[0] > 0) * np.fabs(flow[0]) * 10
        umap = np.clip(umap, 0, 255).astype(np.uint8)
        vmap[:, :, 2] = (flow[1] < 0) * np.fabs(flow[1]) * 10
        vmap[:, :, 0] = (flow[1] > 0) * np.fabs(flow[1]) * 10
        vmap = np.clip(vmap, 0, 255).astype(np.uint8)

        return umap, vmap


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel
