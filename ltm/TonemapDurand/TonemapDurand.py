import numpy as np
import sys
import cv2

if __name__ == '__main__':
    gamma = 0.7
    contrast = 1.0
    sigma_space = 20
    sigma_color = 20

    #读取hdr图像
    hdr_out_rgb = np.fromfile(sys.argv[1], dtype='float32').reshape(384, 512, 3);
    hdr_out_rgb = cv2.resize(hdr_out_rgb, (192, 256))
    cv2.imshow('hdr_out_rgb', hdr_out_rgb)

    image_gray = cv2.cvtColor(hdr_out_rgb, cv2.COLOR_BGR2GRAY)
    cv2.imshow('hdr_out_gray', image_gray);

    image_intensity_log = image_gray#np.log10(image_gray + eps)

    img_base = cv2.bilateralFilter(image_intensity_log, 7, sigma_color, sigma_space);
    cv2.imshow('img_base', img_base)
   
    image_detail = image_intensity_log - img_base

    dst = gamma*img_base + contrast*image_detail
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
