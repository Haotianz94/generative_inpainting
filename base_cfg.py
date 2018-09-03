import torch
import cv2

base_cfg = {}
# data_loader
base_cfg['input_pool_path'] = '../data_home/input_pool.tar'
base_cfg['video_list'] = ['bmx-bumps',
                     'breakdance',
                     'breakdance-flare',
                     'camel',
                     'car-shadow',
                     'horsejump-high',
                     'parkour',
                     'rhino',
                     'tennis',
                     'train']
base_cfg['mask_list'] = [None]                     
base_cfg['data_dir'] = '../data_home/SYN/'
base_cfg['video_name'] = 'wall_large'
base_cfg['mask_name'] = None
base_cfg['batch_size'] = 3
base_cfg['batch_stride'] = 1
base_cfg['frame_sum'] = 60
base_cfg['resize'] = 1
base_cfg['interpolation'] = cv2.INTER_AREA
base_cfg['input_mode'] = 'dip'
base_cfg['input_type'] = 'noise' # 'mesh_grid'
base_cfg['input_ratio'] = 0.1
base_cfg['warp_input'] = False
base_cfg['dim_div_by'] = 64
base_cfg['dilation_iter'] = 0
base_cfg['batch_mode'] = 'seq'
base_cfg['input_noise_std'] = 0
base_cfg['use_flow'] = False

# model
base_cfg['net_type'] = 'skip' # one of skip_depth4|skip_depth2|UNET|ResNet
base_cfg['net_version'] = '2d' # 2d|3d
base_cfg['net_depth'] = 6
base_cfg['input_channel'] = 1
base_cfg['output_channel'] = 3
base_cfg['num_channels_down'] = [16, 32, 64, 128, 128, 128]
base_cfg['num_channels_up'] = [16, 32, 64, 128, 128, 128]
base_cfg['num_channels_skip'] = [4, 4, 4, 4, 4, 4]
base_cfg['filter_size_down'] = 5 # (3, 5, 5)
base_cfg['filter_size_up'] = 3 # (3, 3, 3)
base_cfg['filter_size_skip'] = 1 # (1, 1, 1)
base_cfg['num_temporal_downsample'] = 0 # (1, 1, 1)
base_cfg['dilation_down'] = [1, 1, 1, 1, 1, 1]
base_cfg['dilation_up'] = [1, 1, 1, 1, 1, 1]
base_cfg['use_skip'] = True
base_cfg['use_cwfc'] = False
base_cfg['pad'] = 'reflection' # 'zero'
base_cfg['dtype'] = torch.cuda.FloatTensor

# loss
base_cfg['loss_weight'] = {'recon_image': 1, 'recon_flow': 0, 'smooth_flow': 0, \
'gt_warp_res': 0, 'res_warp_res_diff_res': 0, 'res_warp_res_diff_gt': 0}

# optimize
base_cfg['LR'] = 1e-2 # skip 1e-2, else 1e-3
base_cfg['OPT_OVER'] = 'net'
base_cfg['param_noise'] = True
base_cfg['num_iter'] = 3000
base_cfg['num_pass'] = 1
base_cfg['regen_input'] = 0
base_cfg['coarse2fine'] = False
base_cfg['pretrain'] = False

# visualize
base_cfg['save_every_iter'] = 300
base_cfg['save_every_pass'] = 1
base_cfg['figsize'] = 15
base_cfg['plot'] = False
base_cfg['save'] = True
base_cfg['draw_boundary'] = True
base_cfg['save_batch'] = True
base_cfg['save_upsample'] = False
base_cfg['compare_gt'] = False

# result
base_cfg['res_dir'] = None