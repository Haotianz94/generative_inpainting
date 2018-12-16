import os
from PIL import Image
import numpy as np
import cv2

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
        self.input_sequence = cfg['input_sequence']
        self.input_reso = cfg.get('input_reso', '')
        self.image_dir = os.path.join(self.data_dir, "Image" + self.input_reso, self.video_name)
        if self.mask_name is None:
            self.mask_dir = os.path.join(self.data_dir, "Mask" + self.input_reso, self.video_name)
        else:
            self.mask_dir = os.path.join(self.data_dir, "Mask" + self.input_reso, self.mask_name)
        if not self.input_sequence:
            self.image_dir += '.avi'
            self.mask_dir += '.avi'
        self.dim_div_by = cfg['dim_div_by']
        self.dilation_iter = cfg['dilation_iter']
        self.batch_mode = cfg['batch_mode']

        # self.input_mode = cfg['input_mode']
        # self.input_type = cfg['input_type']
        # self.input_channel = cfg['input_channel']
        # self.input_ratio = cfg['input_ratio']
        self.warp_input = cfg['warp_input']
        self.input_noise_std = cfg['input_noise_std']
        self.resize = cfg['resize']
        self.batch_stride = cfg['batch_stride']
        
        if self.input_sequence:
            # get true frame sum
            frame_sum_true = 0
            for file in os.listdir(self.image_dir):
                if file[0] != '.':
                    frame_sum_true += 1
            # get frame size
            image_path = os.path.join(self.image_dir, '00000.jpg')
            assert(os.path.exists(image_path))
            img = cv2.imread(image_path)
            self.frame_H, self.frame_W = int(img.shape[0] * self.resize // self.dim_div_by * self.dim_div_by), \
                int(img.shape[1] * self.resize // self.dim_div_by * self.dim_div_by)
        else:
            cap = cv2.VideoCapture(self.image_dir)
            frame_sum_true = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.resize // self.dim_div_by * self.dim_div_by)
            self.frame_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.resize // self.dim_div_by * self.dim_div_by)
            cap.release()
        

        self.frame_sum = min(cfg.get('frame_sum_max', 200), frame_sum_true)
        self.cfg['frame_sum'] = self.frame_sum = self.frame_sum // (2**cfg['num_temporal_downsample']) * (2**cfg['num_temporal_downsample'])
        cfg['batch_size'] = self.batch_size = min(cfg['batch_size'], self.frame_sum)
        self.frame_size = (self.frame_H, self.frame_W, 3) 
        cfg['frame_size'] = self.frame_size# = self.frame_size[::-1]
        
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
        if not self.input_sequence:
            cap_image = cv2.VideoCapture(self.image_dir)
            cap_mask = cv2.VideoCapture(self.mask_dir)
        for fid in range(self.frame_sum):
            if self.input_sequence:
                frame, mask = self.load_single_frame(fid)
            else:
                frame, mask = self.load_single_frame_v(cap_image, cap_mask)
            _, contour, hier = cv2.findContours(mask[:, :, 0].astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            self.image_all.append(frame)
            self.mask_all.append(mask)
            self.contour_all.append(contour)
        if not self.input_sequence:
            cap_image.release()
            cap_mask.release()


    def init_batch_list(self):
        self.batch_list = []
        # for i in range(0, self.frame_sum - self.batch_size * self.batch_stride+1, self.traverse_step):
        #     for j in range(self.batch_stride):
        #         self.batch_list.append(i + j)
        for i in range(0, self.frame_sum - (self.batch_size - 1) * self.batch_stride, self.traverse_step):
            self.batch_list.append(i)

        if self.batch_mode == 'seq':
            if self.reverse:
                self.batch_list = self.batch_list[::-1]
        elif self.batch_mode == 'random':
            median = self.batch_list[len(self.batch_list) // 2]
            random.shuffle(self.batch_list)
            self.batch_list.remove(median)
            self.batch_list.append(median)

    
    def next_batch(self):
        if len(self.batch_list) == 0:
            self.reverse = not self.reverse
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
            ###
            if self.cfg['use_square_mask']:
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
                    # mask_square[self.frame_H//4*3 : ymax+1, xmin : xmax+1, :] = np.array([255, 255, 255])
                    # mask_square[ymin : ymax+1, self.frame_W//2-20 : xmax+1, :] = np.array([255, 255, 255])
                mask = mask_square
            ###
            image = gt * (1. - (mask > 0)) + mask
            input = np.concatenate([image.astype('uint8'), mask], axis=1)
            img = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            mask = mask[:, :, 0:1]

            img_batch.append(img)
            mask_batch.append(mask)
            input_batch.append(input) 
            contour_batch.append(contour)           

        batch_data = {'batch_idx': batch_idx, 'input_batch': np.array(input_batch), \
        'img_batch': np.array(img_batch), 'mask_batch': np.array(mask_batch), 'contour_batch': contour_batch}
        return batch_data

    
    def load_single_frame(self, fid):
        image_path = os.path.join(self.image_dir, '{:05}'.format(fid) + '.jpg')
        if not os.path.exists(image_path):
            image_path = image_path.replace('jpg', 'png')
        mask_path = os.path.join(self.mask_dir, '{:05}'.format(fid) + '.png')
        gt = self.load_image(image_path)
        mask = self.load_image(mask_path)
        return gt, mask

    
    def load_image(self, path):
        img = cv2.imread(path)
        img = self.warp_resize_crop(img, self.resize)
        return img.astype('uint8')


    def load_single_frame_v(self, cap_image, cap_mask):
        gt = self.load_image_v(cap_image, False)
        mask = self.load_image_v(cap_mask, True)
        return gt, mask

    
    def load_image_v(self, cap, is_mask):
        _, img = cap.read()
        img = self.warp_resize_crop(img, self.resize)
        if is_mask:
            img = (img > 0) * 255
            img = img.astype('uint8')

        return img
    
    
    def warp_resize_crop(self, img, scale):
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
    