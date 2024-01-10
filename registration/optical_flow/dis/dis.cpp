#include "dis.hpp"

#define DEBUG 1

MyDis::MyDis() {
    finest_scale = 2;
    coarsest_scale = 10;
    patch_size = 8;
    patch_stride = 4;
    grad_descent_iter = 16;
    variational_refinement_iter = 5;
    variational_refinement_alpha = 20.f;
    variational_refinement_gamma = 10.f;
    variational_refinement_delta = 5.f;
    border_size = 16;
}

MyDis::~MyDis() {
}

template <typename T> static inline void spatialGradientKernel( T& vx, T& vy,
        const T& v00, const T& v01, const T& v02,
        const T& v10,               const T& v12,
        const T& v20, const T& v21, const T& v22 ) {
    T tmp_add = v22 - v00,
      tmp_sub = v02 - v20,
      tmp_x   = v12 - v10,
      tmp_y   = v21 - v01;

    vx = tmp_add + tmp_sub + tmp_x + tmp_x;
    vy = tmp_add - tmp_sub + tmp_y + tmp_y;
}

void spatialGradient( InputArray _src, OutputArray _dx, OutputArray _dy,
        int ksize, int borderType ) {
    // Prepare InputArray src
    Mat src = _src.getMat();

    // Prepare OutputArrays dx, dy
    _dx.create( src.size(), CV_16SC1 );
    _dy.create( src.size(), CV_16SC1 );
    Mat dx = _dx.getMat(), dy = _dy.getMat();

    // Get dimensions
    const int H = src.rows, W = src.cols;

    // Row, column indices
    int i = 0, j = 0;

    // Handle border types
    int i_top    = 0,     // Case for H == 1 && W == 1 && BORDER_REPLICATE
        i_bottom = H - 1,
        j_offl   = 0,     // j offset from 0th   pixel to reach -1st pixel
        j_offr   = 0;     // j offset from W-1th pixel to reach Wth  pixel

    if ( borderType == BORDER_DEFAULT ) {// Equiv. to BORDER_REFLECT_101
        if ( H > 1 ) {
            i_top    = 1;
            i_bottom = H - 2;
        }
        if ( W > 1 ) {
            j_offl = 1;
            j_offr = -1;
        }
    }

    int i_start = 0;
    int j_start = 0;
    int j_p, j_n;
    uchar v00, v01, v02, v10, v11, v12, v20, v21, v22;
    for ( i = 0; i < H; i++ ) {
        uchar *p_src = src.ptr<uchar>(i == 0 ? i_top : i - 1);
        uchar *c_src = src.ptr<uchar>(i);
        uchar *n_src = src.ptr<uchar>(i == H - 1 ? i_bottom : i + 1);

        short *c_dx = dx.ptr<short>(i);
        short *c_dy = dy.ptr<short>(i);

        // Process left-most column
        j = 0;
        j_p = j + j_offl;
        j_n = 1;
        if ( j_n >= W ) j_n = j + j_offr;
        v00 = p_src[j_p]; v01 = p_src[j]; v02 = p_src[j_n];
        v10 = c_src[j_p]; v11 = c_src[j]; v12 = c_src[j_n];
        v20 = n_src[j_p]; v21 = n_src[j]; v22 = n_src[j_n];
        spatialGradientKernel<short>( c_dx[0], c_dy[0], v00, v01, v02, v10,
                v12, v20, v21, v22 );
        v00 = v01; v10 = v11; v20 = v21;
        v01 = v02; v11 = v12; v21 = v22;

        // Process middle columns
        j = i >= i_start ? 1 : j_start;
        j_p = j - 1;
        v00 = p_src[j_p]; v01 = p_src[j];
        v10 = c_src[j_p]; v11 = c_src[j];
        v20 = n_src[j_p]; v21 = n_src[j];

        for ( ; j < W - 1; j++ ) {
            // Get values for next column
            j_n = j + 1; v02 = p_src[j_n]; v12 = c_src[j_n]; v22 = n_src[j_n];
            spatialGradientKernel<short>( c_dx[j], c_dy[j], v00, v01, v02, v10,
                    v12, v20, v21, v22 );

            // Move values back one column for next iteration
            v00 = v01; v10 = v11; v20 = v21;
            v01 = v02; v11 = v12; v21 = v22;
        }

        // Process right-most column
        if ( j < W ) {
            j_n = j + j_offr; v02 = p_src[j_n]; v12 = c_src[j_n]; v22 = n_src[j_n];
            spatialGradientKernel<short>( c_dx[j], c_dy[j], v00, v01, v02, v10,
                    v12, v20, v21, v22 );
        }
    }
}

void MyDis::prepareBuffers(Mat &I0, Mat &I1, Mat &flow) {
    I0s.resize(coarsest_scale + 1);
    I1s.resize(coarsest_scale + 1);
    I1s_ext.resize(coarsest_scale + 1);
    I0xs.resize(coarsest_scale + 1);
    I0ys.resize(coarsest_scale + 1);
    Ux.resize(coarsest_scale + 1);
    Uy.resize(coarsest_scale + 1);

    int fraction = 1;
    int cur_rows = 0, cur_cols = 0;

    for (int i = 0; i <= coarsest_scale; i++) {
        if (i == finest_scale) {
            cur_rows = I0.rows / fraction;
            cur_cols = I0.cols / fraction;
            I0s[i] = Mat::zeros(Size(cur_cols, cur_rows), CV_8UC1);
            resize(I0, I0s[i], I0s[i].size(), 0.0, 0.0, INTER_AREA);
            I1s[i] = Mat::zeros(Size(cur_cols, cur_rows), CV_8UC1);
            resize(I1, I1s[i], I1s[i].size(), 0.0, 0.0, INTER_AREA);

            Sx = Mat::zeros(Size(cur_cols/patch_stride, cur_rows/patch_stride), CV_32FC1);
            Sy = Mat::zeros(Size(cur_cols/patch_stride, cur_rows/patch_stride), CV_32FC1);
            I0xx_buf = Mat::zeros(Size(cur_cols/patch_stride, cur_rows/patch_stride), CV_32FC1);
            I0yy_buf = Mat::zeros(Size(cur_cols/patch_stride, cur_rows/patch_stride), CV_32FC1);
            I0xy_buf = Mat::zeros(Size(cur_cols/patch_stride, cur_rows/patch_stride), CV_32FC1);
            I0x_buf  = Mat::zeros(Size(cur_cols/patch_stride, cur_rows/patch_stride), CV_32FC1);
            I0y_buf  = Mat::zeros(Size(cur_cols/patch_stride, cur_rows/patch_stride), CV_32FC1);

            I0xx_buf_aux = Mat::zeros(Size(cur_cols/patch_stride, cur_rows), CV_32FC1);
            I0yy_buf_aux = Mat::zeros(Size(cur_cols/patch_stride, cur_rows), CV_32FC1);
            I0xy_buf_aux = Mat::zeros(Size(cur_cols/patch_stride, cur_rows), CV_32FC1);
            I0x_buf_aux = Mat::zeros(Size(cur_cols/patch_stride, cur_rows), CV_32FC1);
            I0y_buf_aux = Mat::zeros(Size(cur_cols/patch_stride, cur_rows), CV_32FC1);

            U = Mat::zeros(Size(cur_cols, cur_rows), CV_32FC2);
        } else if (i > finest_scale) {
            cur_rows = I0s[i - 1].rows / 2;
            cur_cols = I0s[i - 1].cols / 2;
            I0s[i] = Mat::zeros(Size(cur_cols, cur_rows), CV_8UC1);
            resize(I0s[i - 1], I0s[i], I0s[i].size(), 0.0, 0.0, INTER_AREA);
            I1s[i] = Mat::zeros(Size(cur_cols, cur_rows), CV_8UC1);
            resize(I1s[i - 1], I1s[i], I1s[i].size(), 0.0, 0.0, INTER_AREA);
        }
        
        if (i >= finest_scale) {
            I1s_ext[i] = Mat::zeros(Size(cur_cols+2*border_size, cur_rows+2*border_size), CV_8UC1);
            copyMakeBorder(I1s[i], I1s_ext[i], border_size, border_size, border_size, border_size, BORDER_REPLICATE);

            I0xs[i] = Mat::zeros(Size(cur_cols, cur_rows), CV_16SC1);
            I0ys[i] = Mat::zeros(Size(cur_cols, cur_rows), CV_16SC1);
            spatialGradient(I0s[i], I0xs[i], I0ys[i]);

            Ux[i] = Mat::zeros(Size(cur_cols, cur_rows), CV_32FC1);
            Uy[i] = Mat::zeros(Size(cur_cols, cur_rows), CV_32FC1);

#if DEBUG
            Mat t1 = abs(I0xs[i]);
            Mat t2 = abs(I0ys[i]);
            t1.convertTo(t1, CV_8UC1);
            t2.convertTo(t2, CV_8UC1);

            imshow("I0xs", t1/4);
            imshow("I0ys", t2/4);
#endif
        }
        
        fraction *= 2;
    }
}

void MyDis::precomputeStructureTensor(Mat &dst_I0xx, Mat &dst_I0yy, Mat &dst_I0xy, Mat &dst_I0x, Mat &dst_I0y, Mat &I0x, Mat &I0y) {
    float *I0xx_ptr = dst_I0xx.ptr<float>();
    float *I0yy_ptr = dst_I0yy.ptr<float>();
    float *I0xy_ptr = dst_I0xy.ptr<float>();
    float *I0x_ptr  = dst_I0x.ptr<float>();
    float *I0y_ptr  = dst_I0y.ptr<float>();

    float *I0xx_aux_ptr = I0xx_buf_aux.ptr<float>();
    float *I0yy_aux_ptr = I0yy_buf_aux.ptr<float>();
    float *I0xy_aux_ptr = I0xy_buf_aux.ptr<float>();
    float *I0x_aux_ptr  = I0x_buf_aux.ptr<float>();
    float *I0y_aux_ptr  = I0y_buf_aux.ptr<float>();

    for (int i = 0; i < h; i++) {
        float sum_xx = 0.0f, sum_yy = 0.0f, sum_xy = 0.0f, sum_x = 0.0f, sum_y = 0.0f;
        short *x_row = I0x.ptr<short>(i);
        short *y_row = I0y.ptr<short>(i);
        for (int j = 0; j < patch_size; j++) {
            sum_xx += x_row[j] * x_row[j];
            sum_yy += y_row[j] * y_row[j];
            sum_xy += x_row[j] * y_row[j];
            sum_x += x_row[j];
            sum_y += y_row[j];
        }
        I0xx_aux_ptr[i * ws] = sum_xx;
        I0yy_aux_ptr[i * ws] = sum_yy;
        I0xy_aux_ptr[i * ws] = sum_xy;
        I0x_aux_ptr[i * ws]  = sum_x;
        I0y_aux_ptr[i * ws]  = sum_y;

        int js = 1;
        for (int j = patch_size; j < w; j++) {
            sum_xx += (x_row[j] * x_row[j] - x_row[j - patch_size] * x_row[j - patch_size]);
            sum_yy += (y_row[j] * y_row[j] - y_row[j - patch_size] * y_row[j - patch_size]);
            sum_xy += (x_row[j] * y_row[j] - x_row[j - patch_size] * y_row[j - patch_size]);
            sum_x += (x_row[j] - x_row[j - patch_size]);
            sum_y += (y_row[j] - y_row[j - patch_size]);
            if ((j - patch_size + 1) % patch_stride == 0) {
                I0xx_aux_ptr[i * ws + js] = sum_xx;
                I0yy_aux_ptr[i * ws + js] = sum_yy;
                I0xy_aux_ptr[i * ws + js] = sum_xy;
                I0x_aux_ptr[i * ws + js] = sum_x;
                I0y_aux_ptr[i * ws + js] = sum_y;
                js++;
            }
        }
    }

    AutoBuffer<float> sum_xx(ws), sum_yy(ws), sum_xy(ws), sum_x(ws), sum_y(ws);
    for (int j = 0; j < ws; j++) {
        sum_xx[j] = 0.0f;
        sum_yy[j] = 0.0f;
        sum_xy[j] = 0.0f;
        sum_x[j] = 0.0f;
        sum_y[j] = 0.0f;
    }

    for (int i = 0; i < patch_size; i++) {
        for (int j = 0; j < ws; j++) {
            sum_xx[j] += I0xx_aux_ptr[i * ws + j];
            sum_yy[j] += I0yy_aux_ptr[i * ws + j];
            sum_xy[j] += I0xy_aux_ptr[i * ws + j];
            sum_x[j] += I0x_aux_ptr[i * ws + j];
            sum_y[j] += I0y_aux_ptr[i * ws + j];
        }
    }

    for (int j = 0; j < ws; j++) {
        I0xx_ptr[j] = sum_xx[j];
        I0yy_ptr[j] = sum_yy[j];
        I0xy_ptr[j] = sum_xy[j];
        I0x_ptr[j] = sum_x[j];
        I0y_ptr[j] = sum_y[j];
    }

    int is = 1;
    for (int i = patch_size; i < h; i++) {
        for (int j = 0; j < ws; j++) {
            sum_xx[j] += (I0xx_aux_ptr[i * ws + j] - I0xx_aux_ptr[(i - patch_size) * ws + j]);
            sum_yy[j] += (I0yy_aux_ptr[i * ws + j] - I0yy_aux_ptr[(i - patch_size) * ws + j]);
            sum_xy[j] += (I0xy_aux_ptr[i * ws + j] - I0xy_aux_ptr[(i - patch_size) * ws + j]);
            sum_x[j] += (I0x_aux_ptr[i * ws + j] - I0x_aux_ptr[(i - patch_size) * ws + j]);
            sum_y[j] += (I0y_aux_ptr[i * ws + j] - I0y_aux_ptr[(i - patch_size) * ws + j]);
        }
        if ((i - patch_size + 1) % patch_stride == 0) {
            for (int j = 0; j < ws; j++) {
                I0xx_ptr[is * ws + j] = sum_xx[j];
                I0yy_ptr[is * ws + j] = sum_yy[j];
                I0xy_ptr[is * ws + j] = sum_xy[j];
                I0x_ptr[is * ws + j] = sum_x[j];
                I0y_ptr[is * ws + j] = sum_y[j];
            }
            is++;
        }
    }
}

void MyDis::PatchInverseSearch_ParBody(int _nstripes, int _hs, Mat &dst_Sx, Mat &dst_Sy,
                                    Mat &src_Ux, Mat &src_Uy, Mat &_I0, Mat &_I1,
                                    Mat &_I0x, Mat &_I0y, int _num_iter, int _pyr_level) {
    int psz = patch_size;
    int psz2 = psz / 2;
    int w_ext = w + 2 * border_size; //!< width of I1_ext
    int bsz = border_size;

    /* Input dense flow */
    float *Ux_ptr = Ux->ptr<float>();
    float *Uy_ptr = Uy->ptr<float>();

    /* Output sparse flow */
    float *Sx_ptr = Sx->ptr<float>();
    float *Sy_ptr = Sy->ptr<float>();

    uchar *I0_ptr = I0->ptr<uchar>();
    uchar *I1_ptr = I1->ptr<uchar>();
    short *I0x_ptr = I0x->ptr<short>();
    short *I0y_ptr = I0y->ptr<short>();

    /* Precomputed structure tensor */
    float *xx_ptr = I0xx_buf.ptr<float>();
    float *yy_ptr = I0yy_buf.ptr<float>();
    float *xy_ptr = I0xy_buf.ptr<float>();
    /* And extra buffers for mean-normalization: */
    float *x_ptr = I0x_buf.ptr<float>();
    float *y_ptr = I0y_buf.ptr<float>();
    float *initial_Ux_ptr = NULL, *initial_Uy_ptr = NULL;

    int i, j, dir;
    int start_is, end_is, start_js, end_js;
    int start_i, start_j;
    float i_lower_limit = bsz - psz + 1.0f;
    float i_upper_limit = bsz + h - 1.0f;
    float j_lower_limit = bsz - psz + 1.0f;
    float j_upper_limit = bsz + w - 1.0f;
    float dUx, dUy, i_I1, j_I1, w00, w01, w10, w11, dx, dy;

#define INIT_BILINEAR_WEIGHTS(Ux, Uy) \
    i_I1 = min(max(i + Uy + bsz, i_lower_limit), i_upper_limit); \
    j_I1 = min(max(j + Ux + bsz, j_lower_limit), j_upper_limit); \
    { \
        float di = i_I1 - floor(i_I1); \
        float dj = j_I1 - floor(j_I1); \
        w11 = di       * dj; \
        w10 = di       * (1 - dj); \
        w01 = (1 - di) * dj; \
        w00 = (1 - di) * (1 - dj); \
    }

#define COMPUTE_SSD(dst, Ux, Uy)                                                                                   \
    INIT_BILINEAR_WEIGHTS(Ux, Uy);                                                                                 \
    dst = computeSSDMeanNorm(I0_ptr + i * w + j, I1_ptr + (int)i_I1 * w_ext + (int)j_I1, w, w_ext, w00,  \
                             w01, w10, w11, psz);                                                                  \

    int num_inner_iter = (int)floor(grad_descent_iter / (float)num_iter);
    for (int iter = 0; iter < num_iter; iter++) {
        if (iter % 2 == 0) {
            dir = 1;
            start_is = min(range.start * stripe_sz, hs);
            end_is = min(range.end * stripe_sz, hs);
            start_js = 0;
            end_js  = ws;
            start_i = start_is * patch_stride;
            start_j = 0;
        } else {
            dir = -1;
            start_is = min(range.end * stripe_sz, hs) - 1;
            end_is = min(range.start * stripe_sz, hs) - 1;
            start_js = ws - 1;
            end_js = -1;
            start_i = start_is * patch_stride;
            start_j = (ws - 1) * patch_stride;
        }

        i = start_i;
        for (int is = start_is; dir * is < dir * end_is; is += dir) {
            j = start_j;
            for (int js = start_js; dir * js < dir * end_js; js += dir) {
                if (iter == 0) {
                    /* Using result form the previous pyramid level as the very first approximation: */
                    Sx_ptr[is * ws + js] = Ux_ptr[(i + psz2) * w + j + psz2];
                    Sy_ptr[is * ws + js] = Uy_ptr[(i + psz2) * w + j + psz2];
                }

                float min_SSD = INF, cur_SSD;
                COMPUTE_SSD(min_SSD, Sx_ptr[is * ws + js], Sy_ptr[is * ws + js]);

                /* Try spatial candidates: */
                if (dir * js > dir * start_js) {
                    COMPUTE_SSD(cur_SSD, Sx_ptr[is * ws + js - dir], Sy_ptr[is * ws + js - dir]);
                    if (cur_SSD < min_SSD) {
                        min_SSD = cur_SSD;
                        Sx_ptr[is * ws + js] = Sx_ptr[is * ws + js - dir];
                        Sy_ptr[is * ws + js] = Sy_ptr[is * ws + js - dir];
                    }
                }
                /* Flow vectors won't actually propagate across different stripes, which is the reason for keeping
                 * the number of stripes constant. It works well enough in practice and doesn't introduce any
                 * visible seams.
                 */
                if (dir * is > dir * start_is) {
                    COMPUTE_SSD(cur_SSD, Sx_ptr[(is - dir) * ws + js], Sy_ptr[(is - dir) * ws + js]);
                    if (cur_SSD < min_SSD) {
                        min_SSD = cur_SSD;
                        Sx_ptr[is * ws + js] = Sx_ptr[(is - dir) * ws + js];
                        Sy_ptr[is * ws + js] = Sy_ptr[(is - dir) * ws + js];
                    }
                }

                /* Use the best candidate as a starting point for the gradient descent: */
                float cur_Ux = Sx_ptr[is * ws + js];
                float cur_Uy = Sy_ptr[is * ws + js];

                /* Computing the inverse of the structure tensor: */
                float detH = xx_ptr[is * ws + js] * yy_ptr[is * ws + js] -
                             xy_ptr[is * ws + js] * xy_ptr[is * ws + js];
                if (abs(detH) < EPS)
                    detH = EPS;
                float invH11 = yy_ptr[is * ws + js] / detH;
                float invH12 = -xy_ptr[is * ws + js] / detH;
                float invH22 = xx_ptr[is * ws + js] / detH;
                float prev_SSD = INF, SSD;
                float x_grad_sum = x_ptr[is * ws + js];
                float y_grad_sum = y_ptr[is * ws + js];

                for (int t = 0; t < num_inner_iter; t++) {
                    INIT_BILINEAR_WEIGHTS(cur_Ux, cur_Uy);
                    SSD = processPatchMeanNorm(dUx, dUy,
                            I0_ptr  + i * w + j, I1_ptr + (int)i_I1 * w_ext + (int)j_I1,
                            I0x_ptr + i * w + j, I0y_ptr + i * w + j,
                            w, w_ext, w00, w01, w10, w11, psz,
                            x_grad_sum, y_grad_sum);

                    dx = invH11 * dUx + invH12 * dUy;
                    dy = invH12 * dUx + invH22 * dUy;
                    cur_Ux -= dx;
                    cur_Uy -= dy;

                    /* Break when patch distance stops decreasing */
                    if (SSD >= prev_SSD)
                        break;
                    prev_SSD = SSD;
                }

                /* If gradient descent converged to a flow vector that is very far from the initial approximation
                 * (more than patch size) then we don't use it. Noticeably improves the robustness.
                 */
                if (norm(Vec2f(cur_Ux - Sx_ptr[is * ws + js], cur_Uy - Sy_ptr[is * ws + js])) <= psz) {
                    Sx_ptr[is * ws + js] = cur_Ux;
                    Sy_ptr[is * ws + js] = cur_Uy;
                }
                j += dir * patch_stride;
            }
            i += dir * patch_stride;
        }
    }
#undef INIT_BILINEAR_WEIGHTS
#undef COMPUTE_SSD
}

Mat MyDis::run(Mat src1, Mat src2) {
    Mat flow = Mat::zeros(src1.size(), CV_32FC2);

    //根据输入图尺寸和patch_size参数，自适应计算金字塔跟踪层数
    coarsest_scale = min((int)(log(max(src1.cols, src1.rows) / (4.0 * patch_size)) / log(2.0) + 0.5),
                         (int)(log(min(src1.cols, src1.rows) / patch_size) / log(2.0)));
    cout << "coarsest_scale:" << coarsest_scale << endl;

    //创建图像金字塔，并计算输入图每层Ix,Iy等相关预处理操作
    prepareBuffers(src1, src2, flow);

    for(int i=coarsest_scale; i>=finest_scale; i--) {
        w = I0s[i].cols;
        h = I0s[i].rows;
        ws = 1 + (w-patch_size) / patch_stride;
        hs = 1 + (h-patch_size) / patch_stride;

        //根据Ix, Iy梯度信息，计算Ix，Iy, Ix*Ix, Ix*Iy, Iy*Iy在patch_size*patch_size窗口中的梯度累加和。
        precomputeStructureTensor(I0xx_buf, I0yy_buf, I0xy_buf, I0x_buf, I0y_buf, I0xs[i], I0ys[i]);
        
        parallel_for_(Range(0, 8), PatchInverseSearch_ParBody(*this, 8, hs, Sx, Sy, Ux[i], Uy[i], I0s[i],
                    I1s_ext[i], I0xs[i], I0ys[i], 2, i));

        /*
        parallel_for_(Range(0, num_stripes),
                Densification_ParBody(*this, num_stripes, I0s[i].rows, Ux[i], Uy[i], Sx, Sy, I0s[i], I1s[i]));
        if (variational_refinement_iter > 0)
            variational_refinement_processors[i]->calcUV(I0s[i], I1s[i], Ux[i], Uy[i]);

        if(i>finest_scale) {
            resize(Ux[i], Ux[i - 1], Ux[i - 1].size());
            resize(Uy[i], Uy[i - 1], Uy[i - 1].size());
            Ux[i - 1] *= 2;
            Uy[i - 1] *= 2;
        }
        */
    }
/*
    Mat uxy[] = {Ux[finest_scale], Uy[finest_scale]};
    merge(uxy, 2, U);
    resize(U, flow, flow.size());
    flow *= 1 << finest_scale;
*/
    return flow;
}
