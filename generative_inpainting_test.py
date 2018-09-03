import neuralgym as ng
from dataset import Dataset
from inpaint_model import InpaintCAModel
from utils.common_utils import *

import numpy as np
import time
import sys
import os
from PIL import Image
import cv2
import scipy
from scipy import ndimage
import tensorflow as tf

class GenerativeInpaintingTest(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.log = {}
        self.log_file = None
        self.data_loader = None

        
    def create_data_loaders(self):
        self.data_loader = Dataset(self.cfg)
            
        
    def create_model(self):
        self.model = InpaintCAModel()

    
    def infer_batch(self, pass_idx, batch_idx, batch_data):
        # forward
        batch_data['pass_idx'] = pass_idx
        ##
        ##

        # log
        log_str = 'Batch {:05}'.format(batch_idx)
        print(log_str)            
        sys.stdout.flush()
        if not self.log_file is None:
            self.log_file.write(log_str + '\n')
            self.log_file.flush()

        # plot and save
        if (self.cfg['plot'] or self.cfg['save']) and pass_idx % self.cfg['save_every_pass'] == self.cfg['save_every_pass'] - 1:   
            self.plot_and_save(batch_idx, batch_data, '{:03}/final'.format(pass_idx+1), self.cfg['save_upsample'])
        return batch_data['out_img_batch']

    
    def plot_and_save(self, batch_idx, batch_data, subpath, save_upsample=False):
        # load cfg
        batch_size = self.cfg['batch_size']
        batch_stride = self.cfg['batch_stride']
        figsize = self.cfg['figsize']
        save_batch = self.cfg['save_batch']
        traverse_step = self.cfg['traverse_step']

        def save(imgs, subpath, subsubpath):
            res_dir = os.path.join(self.cfg['res_dir'], subpath, subsubpath)
            for i, img in enumerate(imgs):
                if img is None:
                    continue
                batch_path = os.path.join(res_dir, 'batch', '{:03}_{:03}.png'.format(batch_idx, batch_idx + i*batch_stride))
                sequence_path = os.path.join(res_dir, 'sequence', '{:03}.png'.format(batch_idx + i*batch_stride))
                img_pil = np_to_pil(img)
                if save_batch:
                    img_pil.save(batch_path)
                if traverse_step <= batch_size // 2:    
                    if batch_idx < batch_stride or i >= batch_size // 2: 
                        img_pil.save(sequence_path)
                else:
                    img_pil.save(sequence_path)

        # load batch data
        img_batch = batch_data['img_batch']
        mask_batch = batch_data['mask_batch']
        out_img_batch = batch_data['out_img_batch'].copy()
        nonhole_batch = img_batch * (1 - mask_batch)
        stitch_batch = nonhole_batch + out_img_batch * mask_batch

        # add upsample batch
        if save_upsample:
            upscale = 4
            w, h = self.cfg['frame_size'][1] * upscale, self.cfg['frame_size'][0] * upscale
            upsample_batch = []
            for img in out_img_batch:
                img_up = cv2.resize(img.transpose(1, 2, 0), (w, h))
                upsample_batch.append(img_up.transpose(2, 0, 1))
            upsample_batch = np.array(upsample_batch)    

        # draw mask boundary
        if self.cfg['draw_boundary']:
            contour_large_batch = batch_data['contour_batch']
            for i in range(batch_size):
                for con in contour_large_batch[i]:
                    for pt in con:
                        x, y = pt[0]
                        out_img_batch[i][y, x, :] = [255, 0, 0]
                        stitch_batch[i][y, x, :] = [255, 0, 0]
                        if save_upsample:
                            upsample_batch[i][y*upscale : y*upscale+2, x*upscale : x*upscale+2, :] = np.repeat([1, 0, 0], 4).reshape(3, 2, 2)

        # store images
        if not self.cfg['res_dir'] is None and self.cfg['save']:
            save(out_img_batch, subpath, 'full')
            if save_upsample:
                save(upsample_batch, subpath, 'upsample')
            if self.cfg['use_flow']:
                if 'f1' in self.cfg['flow_type']:    
                    save(vis_flow_f1_hsv, subpath, 'flow_f1_hsv')
                    save(vis_flow_f1_ycbcr, subpath, 'flow_f1_ycbcr')
                    # save(vis_flow_f1_umap, subpath, 'flow_f1_umap')
                    # save(vis_flow_f1_vmap, subpath, 'flow_f1_vmap')
                    if 'warped_img_f1' in batch_data:              
                        # save(warped_img_f1, subpath, 'warped_img_f1')
                        save(warped_diff_f1, subpath, 'warped_diff_f1')
                if 'b1' in self.cfg['flow_type']:          
                    save(vis_flow_b1_hsv, subpath, 'flow_b1_hsv')
                    save(vis_flow_b1_ycbcr, subpath, 'flow_b1_ycbcr')
                    # save(vis_flow_b1_umap, subpath, 'flow_b1_umap')
                    # save(vis_flow_b1_vmap, subpath, 'flow_b1_vmap')
                    if 'warped_img_b1' in batch_data:
                        # save(warped_img_b1, subpath, 'warped_img_b1')
                        save(warped_diff_b1, subpath, 'warped_diff_b1')
#             save(diff_np, subpath, 'diff')
    

    def build_dir(self, res_dir, subpath):
        os.mkdir(os.path.join(res_dir, subpath))
        res_type_list = ['full']
        if self.cfg['save_upsample']:
            res_type_list += ['upsample']
        if self.cfg['use_flow']:
            if 'f1' in self.cfg['flow_type']:    
                res_type_list += ['flow_f1_hsv', 'flow_f1_ycbcr', 'flow_f1_umap', 'flow_f1_vmap', 'warped_img_f1', 'warped_diff_f1']
            if 'b1' in self.cfg['flow_type']:
                res_type_list += ['flow_b1_hsv', 'flow_b1_ycbcr', 'flow_b1_umap', 'flow_b1_vmap', 'warped_img_b1', 'warped_diff_b1']
        for res_type in res_type_list:
            sub_res_dir = os.path.join(res_dir, subpath, res_type)
            os.mkdir(sub_res_dir)
            os.mkdir(os.path.join(sub_res_dir, 'sequence'))
            if self.cfg['save_batch']:
                os.mkdir(os.path.join(sub_res_dir, 'batch'))

    
    def train(self, sleep=False):
        # randomly sleep seconds to avoid confilctc
        if sleep:
            t = np.random.randint(1, 100)
            time.sleep(t)


        # start time
        start_time = time.time()
        self.log['cfg'] = self.cfg.copy()
        
        # build result folder
        res_dir = self.cfg['res_dir']
        if not res_dir is None:
            if not os.path.exists(res_dir):
                os.mkdir(res_dir)
            # video folder 
            if not self.cfg['mask_name'] is None:   
                res_dir = os.path.join(res_dir, self.cfg['video_name'] + '_' + self.cfg['mask_name'])
            else:
                res_dir = os.path.join(res_dir, self.cfg['video_name'])
            self.cfg['res_dir'] = res_dir
            if os.path.exists(res_dir):
                # exit
                print("Video folder existed!!!")
                return
            
            os.mkdir(res_dir)
            # pass folder
            res_dir = os.path.join(self.cfg['res_dir'], '{:03}'.format(1))
            os.mkdir(res_dir)
            self.build_dir(res_dir, 'final')

            self.log_file = open(os.path.join(self.cfg['res_dir'], 'log.txt'), 'w')
            for key in sorted(self.cfg):
                self.log_file.write(key + ' ' + str(self.cfg[key]) + '\n')

        # print cfg to console       
        for key in sorted(self.cfg):
            print(key, self.cfg[key])

        self.create_data_loaders()
        self.infer(0)
        
        # save time
        running_time = time.time() - start_time
        print("Running time: ", running_time)
        if not self.cfg['res_dir'] is None: 
            self.log_file.write("Running time: " + str(running_time) + '\n')
            self.log_file.close()

    
    def infer(self, pass_idx):
        # load cfg
        batch_size = self.cfg['batch_size']
        print('Pass %d infer start...' % (pass_idx))
        if not self.log_file is None:
            self.log_file.write('Pass ' + str(pass_idx) + 'infer start...\n')
        self.data_loader.set_mode('infer')
        
        while True:
            batch_data = self.data_loader.next_batch()
            if batch_data is None:
                break
            batch_idx = batch_data['batch_idx']
            self.infer_batch(batch_idx, batch_data)    


    def infer_batch(self, batch_idx, batch_data):
        self.create_model()
        tf.reset_default_graph()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            
            input_image = batch_data['input_batch']   

            input_image = tf.constant(input_image, dtype=tf.float32)
            output = self.model.build_server_graph(input_image)
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)

            # load pretrained model
            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            for var in vars_list:
                vname = var.name
                from_name = vname
                var_value = tf.contrib.framework.load_variable(self.cfg['checkpoint_dir'], from_name)
                assign_ops.append(tf.assign(var, var_value))
            sess.run(assign_ops)
            # print('Model loaded.')

            result = sess.run(output)
            out_img_batch = result
            batch_data['out_img_batch'] = out_img_batch

            self.plot_and_save(batch_idx, batch_data, '{:03}/final'.format(1), self.cfg['save_upsample'])

        # log
        log_str = 'Batch {:05}'.format(batch_idx)
        print(log_str)            
        sys.stdout.flush()
        if not self.log_file is None:
            self.log_file.write(log_str + '\n')
            self.log_file.flush()

