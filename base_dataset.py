import os
from PIL import Image
import numpy as np
import cv2

class Dataset(object):
    """
    Simple version of inpainting_dataset 
    Note:
    1. Need to change BGR to RGB
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_dir = cfg['data_dir']
        self.video_name = cfg['video_name']
        self.mask_name = cfg.get('mask_name', None)
        self.data_type = cfg['data_type']
        self.image_dir = os.path.join(self.data_dir, "Image_{}{}".format(self.data_type, cfg['data_resolution']), self.video_name)
        if self.mask_name is None:
            self.mask_dir = os.path.join(self.data_dir, "Mask_{}{}".format(self.data_type, cfg['data_resolution']), self.video_name)
        else:
            self.mask_dir = os.path.join(self.data_dir, "Mask_{}{}".format(self.data_type, cfg['data_resolution']), self.mask_name)
        if self.data_type == 'video':
            self.image_dir += '.avi'
            if not os.path.exists(self.image_dir):
                self.image_dir = self.image_dir.replace('avi', 'mp4')
            self.mask_dir += '.avi'
            if not os.path.exists(self.mask_dir):
                self.mask_dir = self.mask_dir.replace('avi', 'mp4')
        self.dim_div_by = cfg.get('dim_div_by', 64)
        self.resize = cfg['resize']
        self.batch_stride = cfg['batch_stride']
        self.traverse_step = cfg['traverse_step']
        self.batch_size = cfg['batch_size']

        
        if self.data_type == 'sequence':
            # get true frame sum
            frame_sum_true = 0
            self.image_name_list = [os.path.join(self.image_dir, filename) for filename in os.listdir(self.image_dir) if filename[0] != '.']
            self.mask_name_list = [os.path.join(self.mask_dir, filename) for filename in os.listdir(self.mask_dir) if filename[0] != '.']
            self.image_name_list.sort()
            self.mask_name_list.sort()
            frame_sum_true = len(self.image_name_list)
            # get frame size
            img = cv2.imread(self.image_name_list[0])
            self.frame_H, self.frame_W = img.shape[0], img.shape[1]
        else:
            cap = cv2.VideoCapture(self.image_dir)
            frame_sum_true = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        

        cfg['frame_sum'] = self.frame_sum = min(cfg.get('frame_sum_max', 300), frame_sum_true)
        cfg['frame_size'] = self.frame_size = (self.frame_H, self.frame_W, 3) 
        
        self.init_frame_mask()
        self.init_batch_list()

        print("Video name: ", self.video_name)
        print("Mask name: ", self.mask_name)
        print("Frame sum: ", self.frame_sum)
        print("Batch size: ", self.batch_size)
        print("Batch stride: ", self.batch_stride)
        print("Frame size: ", self.frame_size)
        print("Resize: ", self.resize)
        print("Interpolation: ", self.cfg['interpolation'])
        print("Dim_div_by: ", self.dim_div_by)


    def init_frame_mask(self):
        self.image_all = []
        self.mask_all = []
        self.contour_all = []
        if self.data_type == 'video':
            cap_image = cv2.VideoCapture(self.image_dir)
            cap_mask = cv2.VideoCapture(self.mask_dir)
        for fid in range(self.frame_sum):
            if self.data_type == 'sequence':
                frame, mask = self.load_single_frame(fid)
            else:
                frame, mask = self.load_single_frame_v(cap_image, cap_mask)
            contour, hier = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            self.image_all.append(frame)
            self.mask_all.append(mask)
            self.contour_all.append(contour)
        if not self.data_type == 'video':
            cap_image.release()
            cap_mask.release()


    def init_batch_list(self):
        self.batch_list = []
        for i in range(0, self.frame_sum - (self.batch_size - 1) * self.batch_stride, self.traverse_step):
            self.batch_list.append(i)

    
    def next_batch(self):
        if len(self.batch_list) == 0:
            self.init_batch_list()
            return None
        else:
            batch_idx = self.batch_list[0]
            self.batch_list.remove(batch_idx)
            cur_batch = range(batch_idx, batch_idx + self.batch_size*self.batch_stride, self.batch_stride)
            return self.get_batch_data(batch_idx, cur_batch)
        
        
    def get_batch_data(self, batch_idx, cur_batch):
        input_batch, img_batch, mask_batch, contour_batch = [], [], [], []
        for i, fid in enumerate(cur_batch):
            gt, mask, contour = self.image_all[fid], self.mask_all[fid], self.contour_all[fid]

            mask_square = np.zeros((self.frame_size), dtype='uint8')
            for con in contour:
                xmin = self.frame_W
                xmax = 0
                ymin = self.frame_H
                ymax = 0
                for pt in con:
                    x, y = pt[0]
                    xmin = min(xmin, x)
                    xmax = max(xmax, x)
                    ymin = min(ymin, y)
                    ymax = max(ymax, y)
                mask_square[ymin : ymax+1, xmin : xmax+1, :] = np.array([255, 255, 255])
            
            image = gt * (1. - (mask_square > 0)) + mask_square
            input = np.concatenate([image.astype('uint8'), mask_square], axis=1)
            img = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            mask = mask[..., None] > 0

            img_batch.append(img)
            mask_batch.append(mask)
            input_batch.append(input) 
            contour_batch.append(contour)                   

        batch_data = {'batch_idx': batch_idx, 'input_batch': np.array(input_batch), \
        'img_batch': np.array(img_batch), 'mask_batch': np.array(mask_batch), 'contour_batch': contour_batch}
        return batch_data

    
    def load_single_frame(self, fid):
        gt = self.load_image(self.image_name_list[fid], False)
        mask = self.load_image(self.mask_name_list[fid], True)
        return gt, mask

    
    def load_image(self, path, is_mask):
        img = cv2.imread(path)
        img = self.resize_image(img, self.resize)
        if is_mask:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = (img > 127) * 255
            img = img.astype('uint8')
        return img


    def load_single_frame_v(self, cap_image, cap_mask):
        gt = self.load_image_v(cap_image, False)
        mask = self.load_image_v(cap_mask, True)
        return gt, mask

    
    def load_image_v(self, cap, is_mask):
        _, img = cap.read()
        img = self.resize_image(img, self.resize)
        if is_mask:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = (img > 127) * 255
            img = img.astype('uint8')
        return img
    
    
    def resize_image(self, img, resize):
        if not resize is None:
            h, w = img.shape[:2]
            source = 1. * h / w
            target = 1. * resize[0] / resize[1]
            if source > target:
                margin = int((h - w * target) // 2)
                img = img[margin:h-margin]
            elif source < target:
                margin = int((w - h / target) // 2)
                img = img[:, margin:w-margin]
            img = cv2.resize(img, (resize[1], resize[0]), interpolation=self.cfg['interpolation'])
        return img
    
