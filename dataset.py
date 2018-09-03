import os
import csv
from PIL import Image
import scipy
from scipy import ndimage
import numpy as np
import pickle
import cv2
from utils.common_utils import *

'''
Data loader for Davis dataset 
'''
class Dataset(object):
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.reverse = False
        self.use_gt = False
        # load cfg
        self.traverse_step = cfg['traverse_step']
        self.data_dir = cfg['data_dir']
        self.video_name = cfg['video_name']
        self.mask_name = cfg.get('mask_name', None)
        if self.mask_name is None:
            self.cfg['mask_name'] = self.mask_name = self.video_name
        self.image_dir = os.path.join(self.data_dir, "Image", self.video_name)
        self.mask_dir = os.path.join(self.data_dir, "Mask", self.mask_name)
        self.flow_dir = os.path.join(self.data_dir, "Flow", self.video_name)
        self.dim_div_by = cfg['dim_div_by']
        self.dilation_iter = cfg['dilation_iter']
        self.input_mode = cfg['input_mode']
        self.input_type = cfg['input_type']
        self.input_channel = cfg['input_channel']
        self.input_ratio = cfg['input_ratio']
        self.batch_mode = cfg['batch_mode']
        self.warp_input = cfg['warp_input']
        self.input_noise_std = cfg['input_noise_std']
        self.use_flow = cfg['use_flow']
        self.resize = cfg['resize']
        self.batch_stride = cfg['batch_stride']
        ### for dil
        self.patch_size = cfg.get('patch_size', 8)
        self.patch_half = self.patch_size // 2
        self.hole_step = cfg.get('hole_step', 20)
        self.use_consistent_hole = cfg.get('use_consistent_hole', True)
        self.use_same_noise = cfg.get('use_same_noise', False)
        self.use_crop = cfg.get('use_crop', False)
        ###
        
        # get true frame sum
        frame_sum_true = 0
        for file in os.listdir(self.image_dir):
            if file[0] != '.':
                frame_sum_true += 1
        self.frame_sum = min(cfg['frame_sum'], frame_sum_true)
        self.cfg['frame_sum'] = self.frame_sum = self.frame_sum // (2**cfg['num_temporal_downsample']) * (2**cfg['num_temporal_downsample'])
        self.cfg['batch_size'] = self.batch_size = min(cfg['batch_size'], self.frame_sum)

        # get frame size
        image_path = os.path.join(self.image_dir, '00000.jpg')
        if not os.path.exists(image_path):
            image_path = image_path.replace('jpg', 'png')
        assert(os.path.exists(image_path))
        img = cv2.imread(image_path)
        self.cfg['frame_size'] = self.frame_size = (int(img.shape[0] * self.resize // self.dim_div_by * self.dim_div_by), \
            int(img.shape[1] * self.resize // self.dim_div_by * self.dim_div_by), 3)
        self.frame_H = self.frame_size[0]
        self.frame_W = self.frame_size[1]
        
        self.train_list_full = []
        for i in range(0, self.frame_sum - (self.batch_size-1)*self.batch_stride, self.traverse_step):
            self.train_list_full.append(i)
        self.train_list = self.train_list_full.copy()
        
        print("Video name: ", self.video_name)
        print("Mask name: ", self.mask_name)
        print("Frame sum: ", self.frame_sum)
        print("Batch size: ", self.batch_size)
        print("Batch stride: ", self.batch_stride)
        print("Frame size: ", self.frame_size)
        print("Resize: ", self.resize)
        print("Interpolation: ", self.cfg['interpolation'])
        print("Dim_div_by: ", self.dim_div_by)


    def set_mode(self, mode):
        if mode == 'infer':
            self.batch_mode_save = self.batch_mode
            self.batch_size_save = self.batch_size
            self.reverse_save = self.reverse
            self.train_list_save = self.train_list.copy()
            
            self.batch_mode = 'seq'
            self.reverse = False
        
        elif mode == 'train':
            self.batch_mode = self.batch_mode_save
            self.batch_size = self.batch_size_save
            self.reverse = self.reverse_save
            self.train_list = self.train_list_save

    
    def next_batch(self):
        
        if self.batch_mode == 'seq':
            if len(self.train_list) == 0:
                self.train_list = self.train_list_full.copy()
                self.reverse = not self.reverse
                return None
            else:
                batch_idx = self.train_list[0] if not self.reverse else self.train_list[-1]
                self.train_list.remove(batch_idx)
                cur_batch = range(batch_idx, batch_idx + self.batch_size*self.batch_stride, self.batch_stride)
        elif self.batch_mode == 'random':
            if len(self.train_list) == 0:
                self.train_list = self.train_list_full.copy()
                return None
            else:
                batch_idx = np.random.choice(self.train_list)
                self.train_list.remove(batch_idx)
                cur_batch = range(batch_idx, batch_idx + self.batch_size*self.batch_stride, self.batch_stride)

        return self.get_batch_data(batch_idx, cur_batch)
        
        
    def get_batch_data(self, batch_idx, cur_batch):
        input_batch, img_batch, mask_batch, contour_batch = [], [], [], []
        for i, fid in enumerate(cur_batch):
            gt, mask = self.load_single_frame(fid)
            image = gt * (1. - (mask > 0)) + mask
            input = np.concatenate([image.astype('uint8'), mask], axis=1)

            img = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            mask = mask[:, :, 0:1]
            _, contour, hier = cv2.findContours(mask[:, :, 0].astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            img_batch.append(img)
            mask_batch.append(mask)
            input_batch.append(input) 
            contour_batch.append(contour)           

        batch_data = {'batch_idx': batch_idx, 'input_batch': np.array(input_batch), \
        'img_batch': np.array(img_batch), 'mask_batch': np.array(mask_batch), 'contour_batch': contour_batch}
        return batch_data

    
    def load_single_frame(self, fid, convert=True):
        image_path = os.path.join(self.image_dir, '{:05}'.format(fid) + '.jpg')
        if not os.path.exists(image_path):
            image_path = image_path.replace('jpg', 'png')
        mask_path = os.path.join(self.mask_dir, '{:05}'.format(fid) + '.png')
        gt = self.load_image(image_path)
        mask = self.load_image(mask_path)
        return gt, mask

    
    def load_image(self, path):
        img = cv2.imread(path)
        img = self.warp_scale_crop(img, self.resize)
        return img.astype('uint8')
    
    
    def warp_scale_crop(self, img, scale):
        if scale != 1:
            # Todo: a better way maybe directly resize to a size divided by 64
            w, h = int(img.shape[1] * scale), int(img.shape[0] * scale)
            img = cv2.resize(img, (w, h), interpolation=self.cfg['interpolation'])
            
        d = self.dim_div_by    
        # Make dimensions divisible by d
        new_size = (img.shape[0] - img.shape[0] % d, 
                    img.shape[1] - img.shape[1] % d)
        bbox = [
                int((img.shape[0] - new_size[0])/2), 
                int((img.shape[1] - new_size[1])/2),
                int((img.shape[0] + new_size[0])/2),
                int((img.shape[1] + new_size[1])/2),
        ]
        img_cropped = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        return img_cropped
    
