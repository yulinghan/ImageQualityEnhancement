import cv2
import wls

src = cv2.imread('../../../data/input/1.png', cv2.IMREAD_GRAYSCALE)
dst = wls.wls_filter(src)

cv2.imshow("src", src)
cv2.imshow("dst", dst/255)
cv2.waitKey(0)
