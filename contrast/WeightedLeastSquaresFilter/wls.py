import cv2
import numpy as np

import scipy.sparse as sparse
import scipy.sparse.linalg as sl


def process_difference_operator(difference_operator, lambda_, alpha, epsilon):
    difference_operator = -lambda_ / (epsilon + (np.absolute(difference_operator)**alpha))
    return difference_operator


def wls_filter(L, lambda_=0.35, alpha=1.2, epsilon=1e-4):
    # Get log-luminance
    L_log = np.log(L.astype(np.float64) + 1e-10)

    # Compute the forward and backward differences of the luminance channel
    dx_forward = L_log - cv2.copyMakeBorder(L_log[:,1:], top=0, bottom=0, left=0, right=1, 
                                        borderType=cv2.BORDER_REPLICATE)
    dx_backward = L_log - cv2.copyMakeBorder(L_log[:,:-1], top=0, bottom=0, left=1, right=0,
                                        borderType=cv2.BORDER_REPLICATE)
    dy_forward = L_log - cv2.copyMakeBorder(L_log[1:,:], top=0, bottom=1, left=0, right=0,
                                        borderType=cv2.BORDER_REPLICATE)
    dy_backward = L_log - cv2.copyMakeBorder(L_log[:-1,:], top=1, bottom=0, left=0, right=0,
                                        borderType=cv2.BORDER_REPLICATE)

    # Weight each derivative
    dx_forward_weighted = process_difference_operator(dx_forward, lambda_, alpha, epsilon)
    dx_forward_weighted[:,-1] = 0

    dx_backward_weighted = process_difference_operator(dx_backward, lambda_, alpha, epsilon)
    dx_backward_weighted[:,0] = 0

    dy_forward_weighted = process_difference_operator(dy_forward, lambda_, alpha, epsilon)
    dy_forward_weighted[-1,:] = 0

    dy_backward_weighted = process_difference_operator(dy_backward, lambda_, alpha, epsilon)
    dy_backward_weighted[0,:] = 0

    central_element = np.ones_like(dx_forward)-(dx_forward_weighted + dx_backward_weighted +
                                   dy_forward_weighted + dy_backward_weighted)

    # Form sparse matrix
    N = L.size
    C = L.shape[1]

    row = np.zeros(5*N)
    col = np.zeros_like(row)
    data = np.zeros_like(row)

    # Central element
    row[:N] = np.arange(N)
    col[:N] = row[:N]
    data[:N] = central_element.ravel()

    # dx_forward
    row[N:2*N] = np.arange(N)
    col[N:2*N] = row[N:2*N] + 1
    data[N:2*N] = dx_forward_weighted.ravel()

    # dx_backward
    row[2*N:3*N] = np.arange(N)
    col[2*N:3*N] = row[2*N:3*N] - 1
    data[2*N:3*N] = dx_backward_weighted.ravel()

    #dy_forward
    row[3*N:4*N] = np.arange(N)
    col[3*N:4*N] = row[3*N:4*N] + C
    data[3*N:4*N] = dy_forward_weighted.ravel()

    #dy_backward
    row[4*N:5*N] = np.arange(N)
    col[4*N:5*N] = row[4*N:5*N] - C
    data[4*N:5*N] = dy_backward_weighted.ravel()

    # Prevent out-of-bounds indices. Overlapping elements sum together, so setting all
    # out-of-bounds values to zero and repositioning them to (0,0) will have no effect
    # on other values in the sparse matrix.
    data[col >= N] = 0
    data[col < 0] = 0
    row[col >= N] = 0
    row[col < 0] = 0
    col[col >= N] = 0
    col[col < 0] = 0

    A = sparse.coo_matrix((data, (row, col))).tocsr()
    b = L.ravel()

    x, info = sl.cg(A=A, b=b)

    x = x.reshape(L.shape)

    return x
