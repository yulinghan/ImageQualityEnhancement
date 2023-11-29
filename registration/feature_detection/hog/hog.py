import cv2
import sys
import numpy as np
import math

def cal_image_grad(src):
    a  = np.array([-1, 0, 1])
    a = np.reshape(a, [1, 3])
    dx = cv2.filter2D(src, -1, a,   borderType=0)
    dy = cv2.filter2D(src, -1, a.T, borderType=0)

    gradient_magnitude = np.sqrt(dx**2 + dy**2);
    gradient_angle = cv2.phase(abs(dx), abs(dy), angleInDegrees=True)

    return gradient_magnitude, gradient_angle

def get_closest_bins(gradient_angle, bin_size, angle_unit):
    idx = int(gradient_angle / angle_unit)
    mod = gradient_angle % angle_unit

    return idx, (idx + 1) % bin_size, mod

def cell_gradient(cell_magnitude, cell_angle, bin_size, angle_unit):
    orientation_centers = [0] * bin_size
    for i in range(cell_magnitude.shape[0]):
        for j in range(cell_magnitude.shape[1]):
            gradient_strength = cell_magnitude[i][j]
            gradient_angle = cell_angle[i][j]
            min_angle, max_angle, mod = get_closest_bins(gradient_angle, bin_size, angle_unit)
            orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit)))
            orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit))

    return orientation_centers


def cal_grad_hist(gradient_magnitude, gradient_angle, cell_size, bin_size):
    height, width = gradient_magnitude.shape
    angle_unit = 360 / bin_size
    cell_gradient_vector = np.zeros((height // cell_size, width // cell_size, bin_size))

    for i in range(cell_gradient_vector.shape[0]):
        for j in range(cell_gradient_vector.shape[1]):
            cell_magnitude = gradient_magnitude[i * cell_size:(i + 1) * cell_size,
                                                j * cell_size:(j + 1) * cell_size]
            cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size,
                                        j * cell_size:(j + 1) * cell_size]
            
            cell_gradient_vector[i][j] = cell_gradient(cell_magnitude, cell_angle, bin_size, angle_unit)

    return cell_gradient_vector

def render_gradient(image, cell_gradient, cell_size, bin_size):
    cell_width = cell_size / 2
    angle_unit = 360 / bin_size
    max_mag = np.array(cell_gradient).max()

    for x in range(cell_gradient.shape[0]):
        for y in range(cell_gradient.shape[1]):
            cell_grad = cell_gradient[x][y]
            cell_grad /= max_mag
            angle = 0

            for magnitude in cell_grad:
                angle_radian = math.radians(angle)
                x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
                y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))
                x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian))
                y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
                cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                angle += angle_unit

    return image

def cal_hog(cell_gradient_vector):
    hog_vector = []

    #使用滑动窗口
    for i in range(cell_gradient_vector.shape[0] - 1):
        for j in range(cell_gradient_vector.shape[1] - 1):
            #4个cell得到一个block
            block_vector = cell_gradient_vector[i:i+2, j:j+2].reshape(-1, 1)

            #归一化
            sum1 = np.linalg.norm(block_vector)
            for m in range(block_vector.shape[0]):
                block_vector[m] /= sum1

            hog_vector.append(block_vector)

    return hog_vector

if __name__ == "__main__":
    src = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    cv2.imshow('src', src)

    #计算图像梯度幅值和角度
    #梯度dx和dy取绝对值，所以方向只有180度
    src = src.astype(float) / 255.0;
    gradient_magnitude, gradient_angle = cal_image_grad(src)

    #图像将8x8像素合并为一个cell，将180方向均分为9份, 根据梯度方向，合并cell内梯度幅值到9个方向内中
    cell_size = 8;
    bin_size  = 9;
    cell_gradient_vector = cal_grad_hist(gradient_magnitude, gradient_angle, cell_size, bin_size)
    print('cell_gradient_vector:', cell_gradient_vector.shape)

    #显示cell信息
    show_cell = render_gradient(src, cell_gradient_vector, cell_size, bin_size)
    cv2.imshow('show_cell', show_cell)

    #带重叠区域的相邻4个ceil合并并归一化，得到最终hog特征
    hog_vector = cal_hog(cell_gradient_vector);

    cv2.waitKey(0)
