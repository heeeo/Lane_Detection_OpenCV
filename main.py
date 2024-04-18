import os
import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np
import easydict
from PIL import Image
from collections import Counter
from pathlib import Path
from lane_detection_module import *

opt = easydict.EasyDict({'source': "./preprocessing/sample/",
                         'save_dir': "./preprocessing/crop2/"
                         })

def preprocessing(opt):
    print("###### START #####")
    path, save_dir = opt.source, opt.save_dir
    file_list = os.listdir(path)
    
    cnt = 1
    
    for file in file_list:
        src = cv2.imread(path + file)
        src_name = os.path.splitext(file)
        dst = src.copy()
        dst2 = src.copy()
        
        try:   
            w_f_r_img = color_filter(dst)
            
            canny = convert_image(w_f_r_img)
            
            new_lines, _, _ = hough_transform(canny, dst)
                        
            left_fitx, right_fitx = extract_boundary(new_lines)

            pts2 = extend_line(dst, left_fitx, right_fitx)

            _, color_warp = fill_poly(dst2, pts2)

            result2 = trim_lane(dst2, color_warp)
            
            sliding_window(result2, save_dir, src_name)

        except:
            sliding_window(src, save_dir, src_name)
            
        print("{}  {}/{}".format(file, cnt, len(file_list)))
        cnt += 1
    print("##### FINISH #####")

# preprocessing(opt)
         
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--source', type=str, default='./preprocessing/sample/')
#     parser.add_argument('--save-dir', type=str, default='./preprocessing/crop2/')
#     opt = parser.parse_args()
    
#     return opt

def main(opt):
    #preprocessing(**vars(opt))
    preprocessing(opt)


if __name__ == "__main__":
    # opt = parse_opt()
    main(opt)