import sys
import os
import pickle
import time
import datetime
import copy
from generative_inpainting_test import GenerativeInpaintingTest
from base_cfg import base_cfg


def run_test(cfg):
    
    for video_name in cfg['video_list']:
        for mask_name in cfg['mask_list']:
            cfg['video_name'] = video_name
            cfg['mask_name'] = mask_name
            test = GenerativeInpaintingTest(copy.deepcopy(cfg))
            test.train()

def main():
    
    cfg_idx_start = int(sys.argv[1])
    cfg_idx_end = int(sys.argv[2])
    for cfg_idx in range(cfg_idx_start, cfg_idx_end+1):
        
        cfg_add = pickle.load(open('../../tmp/cfg/cfg_' + str(cfg_idx) + '.pkl', 'rb'))
        cfg = copy.deepcopy(base_cfg)
        for key, value in cfg_add.items():
            cfg[key] = value
    
        run_test(cfg)

        
if __name__ == "__main__":
    main()