import numpy as np
import cv2

def display_model(luma, L_peak, L_black, L_refl, gamma):
    luminance = np.power(luma,gamma)*(L_peak - L_black) + L_black + L_refl;
    return luminance

def compute_HDR_gray_histogram(log_gray, bins):
    global_hist, _ = np.histogram(log_gray, bins)
    global_hist = global_hist / global_hist.sum()
    return global_hist


def compute_tone_curve(base_layer_gray_histogram_log, tone_inf_log, tone_sup_log, delta):
    tone_range_log = tone_sup_log - tone_inf_log;
    print('tone_range_log:', tone_range_log, ', delta:', delta)

    p_t = 1e-9
    tmp = float('inf')
    nb_bin = base_layer_gray_histogram_log.shape[0]
    slopes = np.zeros(nb_bin-1, dtype=np.float32)
    omega_t = []
    sum_all = 0

    for iter in range(6):
        sum0 = 0.0

        omega_t = np.where(base_layer_gray_histogram_log>p_t)
        for i in range(len(omega_t[0])):
            sum0 += 1.0 /base_layer_gray_histogram_log[omega_t[0][i]]

        p_t = max(0,(len(omega_t[0]) - tone_range_log/delta)/sum0);
        sum_all = sum0

    for i in range(len(omega_t[0])):
        tmp = base_layer_gray_histogram_log[omega_t[0][i]]*sum_all;
        slopes[omega_t[0][i]] = 1 + (tone_range_log/delta - len(omega_t[0])) / tmp;
    print('slopes:', slopes)

    tone_curve = np.zeros(nb_bin+1, dtype=np.float32)
    tone_curve[0] = 0;
    for i in range(0, slopes.shape[0]):
        tone_curve[i+1] = tone_curve[i] + slopes[i] * delta
    tone_curve[-1] = tone_curve[-2];
    print('tone_curve:', tone_curve)

    xs = np.linspace(0.0, 65535, 65535+1)
    xs[0] = 1.0
    xs = np.log10(xs) / (np.log10(65536)/nb_bin)
    xs_a, xs_b = np.floor(xs).astype(np.uint32), np.ceil(xs).astype(np.uint32)
    alpha = xs - xs_a
    lut = tone_curve[xs_a] * (1 - alpha) + tone_curve[xs_b] * alpha

    return lut

def noise_aware_tone_mapping(gray):
    cv2.imshow('gray', gray*48)

    L_peak = 255;
    L_black = 0.5;
    E_amb = 30;
    gamma = 2.2;
    k = 0.01;
    pi = 3.14159
    L_refl = k*E_amb/pi;

    L_d = display_model([0,1],L_peak,L_black,L_refl,gamma);
    tone_inf_log = np.log10(L_d[0]);
    tone_sup_log = np.log10(L_d[1]);
    print('tone_inf_log:', tone_inf_log, ", tone_sup_log:", tone_sup_log)

    N_iter = 10;
    sigma = 0.1;
    nb_bin = 50;
    gamma_color = 0.6;
    nb_bloc_row = 8;
    nb_bloc_col = 8;
    local_trust = 0.9;

    gray = gray * 65535
    gray = gray.clip(1, 65535)
    log_gray = np.log10(gray)
    log_HDR_luma_absciss = np.linspace(log_gray.min(), log_gray.max(), nb_bin)
    print('log_gray.min:', log_gray.min(), ", log_gray.max():", log_gray.max())
    delta = log_HDR_luma_absciss[1] - log_HDR_luma_absciss[0];
    
    base_layer_gray_histogram_log = compute_HDR_gray_histogram(log_gray, nb_bin)
    global_tone_curve = compute_tone_curve(base_layer_gray_histogram_log,tone_inf_log,tone_sup_log,delta)
    print('global_tone_curve:', global_tone_curve)

    print('log_gray:', log_gray.min(), log_gray.max())
    out = global_tone_curve[gray.astype(np.uint32)]
    print('out1:', out.min(), out.max())
    out = np.power(10, out)
    print('out2:', out.min(), out.max())
    cv2.imshow('out', out/255)
    cv2.waitKey(0)

if __name__ == '__main__':
    hdr_out_rgb_wb = np.fromfile(r'images/hdr_out_rgb_wb.raw', dtype='float64').reshape(2240, 2992, 3)
    img_gray = np.mean(hdr_out_rgb_wb, 2).astype(np.float32)
    img_resize = cv2.resize(img_gray,(0,0), fx = 0.125, fy = 0.125)
    print('hdr_out_rgb_wb2:', img_resize.shape, ', type:', img_resize.dtype)
    noise_aware_tone_mapping(img_resize)
