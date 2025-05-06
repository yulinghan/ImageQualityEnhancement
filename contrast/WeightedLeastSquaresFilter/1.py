import argparse
import cv2
import numpy as np

import wls
 
def parse_args():
    parser = argparse.ArgumentParser(description='Change tone and detail properties of an image')
    parser.add_argument('--img_path', type=str, required=True, help='Path to image')
    args = parser.parse_args()

    return args


def main(args):
    img = cv2.imread(args.img_path)
    cv2.imshow('1', img);

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = img_lab[..., 0]

    img_lab[..., 0] = wls.wls_filter(L, 0.1, 1.5)
    img_lab = np.round(img_lab).astype(np.uint8)
    img_out = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

    cv2.imshow('2', img_out);
    cv2.waitKey(0);

if __name__ == '__main__':
    args = parse_args()
    main(args)
