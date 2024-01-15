import cv2
import numpy as np
import random

def compute_fundamental(x1, x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # build matrix for equations
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [x1[0, i] * x2[0, i], x1[0, i] * x2[1, i], x1[0, i] * x2[2, i],
                x1[1, i] * x2[0, i], x1[1, i] * x2[1, i], x1[1, i] * x2[2, i],
                x1[2, i] * x2[0, i], x1[2, i] * x2[1, i], x1[2, i] * x2[2, i]]

    # compute linear least square solution
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # constrain F
    # make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))

    return F / F[2, 2]


def compute_fundamental_normalized(x1, x2):
    """    Computes the fundamental matrix from corresponding points
        (x1,x2 3*n arrays) using the normalized 8 point algorithm. """

    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2], axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1, 0, -S1 * mean_1[0]], [0, S1, -S1 * mean_1[1]], [0, 0, 1]])
    x1 = np.dot(T1, x1)

    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2], axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2, 0, -S2 * mean_2[0]], [0, S2, -S2 * mean_2[1]], [0, 0, 1]])
    x2 = np.dot(T2, x2)

    # compute F with the normalized coordinates
    F = compute_fundamental(x1, x2)
    # print (F)
    # reverse normalization
    F = np.dot(T1.T, np.dot(F, T2))

    return F / F[2, 2]

def randSeed(good, num = 8):
    '''
    :param good: 初始的匹配点对
    :param num: 选择随机选取的点对数量
    :return: 8个点对list
    '''
    eight_point = random.sample(good, num)
    return eight_point

def PointCoordinates(eight_points, keypoints1, keypoints2):
    '''
    :param eight_points: 随机八点
    :param keypoints1: 点坐标
    :param keypoints2: 点坐标
    :return:8个点
    '''
    x1 = []
    x2 = []
    tuple_dim = (1.,)
    for i in eight_points:
        tuple_x1 = keypoints1[i[0].queryIdx].pt + tuple_dim
        tuple_x2 = keypoints2[i[0].trainIdx].pt + tuple_dim
        x1.append(tuple_x1)
        x2.append(tuple_x2)
    return np.array(x1, dtype=float), np.array(x2, dtype=float)


def ransac(good, keypoints1, keypoints2, confidence,iter_num):
    Max_num = 0
    good_F = np.zeros([3,3])
    inlier_points = []
    for i in range(iter_num):
        eight_points = randSeed(good)
        x1,x2 = PointCoordinates(eight_points, keypoints1, keypoints2)
        F = compute_fundamental_normalized(x1.T, x2.T)
        num, ransac_good = inlier(F, good, keypoints1, keypoints2, confidence)
        if num > Max_num:
            Max_num = num
            good_F = F
            inlier_points = ransac_good
    print(Max_num, good_F)
    return Max_num, good_F, inlier_points


def computeReprojError(x1, x2, F):
    """
    计算投影误差
    """
    ww = 1.0/(F[2,0]*x1[0]+F[2,1]*x1[1]+F[2,2])
    dx = (F[0,0]*x1[0]+F[0,1]*x1[1]+F[0,2])*ww - x2[0]
    dy = (F[1,0]*x1[0]+F[1,1]*x1[1]+F[1,2])*ww - x2[1]
    return dx*dx + dy*dy

def inlier(F,good, keypoints1,keypoints2,confidence):
    num = 0
    ransac_good = []
    x1, x2 = PointCoordinates(good, keypoints1, keypoints2)
    for i in range(len(x2)):
        line = F.dot(x1[i].T)
        #在对极几何中极线表达式为[A B C],Ax+By+C=0,  方向向量可以表示为[-B,A]
        line_v = np.array([-line[1], line[0]])
        err = h = np.linalg.norm(np.cross(x2[i,:2], line_v)/np.linalg.norm(line_v))
        # err = computeReprojError(x1[i], x2[i], F)
        if abs(err) < confidence:
            ransac_good.append(good[i])
            num += 1
    return num, ransac_good


if __name__ =='__main__':
    im1 = 'm1.jpg'
    im2 = 'm2.jpg'

    print(cv2.__version__)
    psd_img_1 = cv2.imread(im1, cv2.IMREAD_COLOR)
    psd_img_2 = cv2.imread(im2, cv2.IMREAD_COLOR)
    # 3) SIFT特征计算
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(psd_img_1, None)
    kp2, des2 = sift.detectAndCompute(psd_img_2, None)

    # FLANN 参数设计
    match = cv2.BFMatcher()
    matches = match.knnMatch(des1, des2, k=2)

    # Apply ratio test
    # 比值测试，首先获取与 A距离最近的点 B （最近）和 C （次近），
    # 只有当 B/C 小于阀值时（0.75）才被认为是匹配，
    # 因为假设匹配是一一对应的，真正的匹配的理想距离为0
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    print(good[0][0])

    print("number of feature points:",len(kp1), len(kp2))
    print(type(kp1[good[0][0].queryIdx].pt))
    print("good match num:{} good match points:".format(len(good)))
    for i in good:
        print(i[0].queryIdx, i[0].trainIdx)


    Max_num, good_F, inlier_points = ransac(good, kp1, kp2, confidence=30, iter_num=500)
    # cv2.drawMatchesKnn expects list of lists as matches.
    # img3 = np.ndarray([2, 2])
    # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:10], img3, flags=2)

    # cv2.drawMatchesKnn expects list of lists as matches.

    img3 = cv2.drawMatchesKnn(psd_img_1,kp1,psd_img_2,kp2,good,None,flags=2)
    img4 = cv2.drawMatchesKnn(psd_img_1,kp1,psd_img_2,kp2,inlier_points,None,flags=2)
    cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
    cv2.imshow("image1",img3)
    cv2.imshow("image2",img4)
    cv2.waitKey(0)#等待按键按下
    cv2.destroyAllWindows()#清除所有窗口