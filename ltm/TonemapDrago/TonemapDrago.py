import numpy as np
import sys
import cv2

def linear_src(src):
    dst = (src - src.min()) / (src.max() - src.min())
    print('hdr max:', src.max(), ", min:", src.min())

    return dst

def log_(src):
    np.clip(src, 1e-4, src.max())
    dst = np.log(src)

    mean = np.exp(np.mean(dst))

    return mean;

def mapLuminance(linaer_src, gray_src, map, saturation):
    b,g,r = cv2.split(linaer_src)
    
    b = np.multiply(b, 1.0/gray_src)
    b = np.power(b, saturation)
    b = np.multiply(b, map)

    g = np.multiply(g, 1.0/gray_src)
    g = np.power(g, saturation)
    g = np.multiply(g, map)

    r = np.multiply(r, 1.0/gray_src)
    r = np.power(r, saturation)
    r = np.multiply(r, map)

    dst = cv2.merge([b, g, r])

    return dst

if __name__ == '__main__':
    bias = 0.85
    saturation = 1.0
    gamma = 1.0

    #读取hdr图像
    hdr_out_rgb = np.fromfile(sys.argv[1], dtype='float32').reshape(384, 512, 3);
    cv2.imshow('hdr_out_rgb', hdr_out_rgb)

    #opencv版本效果
    #tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
    #ldrDrago = tonemapDrago.process(hdr_out_rgb)
    #cv2.imshow('ldrDrago', ldrDrago)

    #hdr图像作归一化
    linaer_src = linear_src(hdr_out_rgb)
    
    #hdr图像转灰度图
    gray_src = cv2.cvtColor(linaer_src, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray_src', gray_src)

    #论文公式实现
    log_mean = log_(gray_src)
    gray_src = gray_src / log_mean
    max_v = gray_src.max()

    map = np.log(gray_src+1.0)
    div = np.power(gray_src/max_v, np.log(bias)/np.log(0.5))
    div = np.log(2.0+8.0*div)
    map = np.multiply(map, 1.0/div)

    dst = mapLuminance(linaer_src, gray_src, map, saturation)
    dst = linear_src(dst)
    cv2.imshow('dst', dst)
    
    cv2.waitKey(0)
