#include "surf_desc.hpp"

MySurfDescTest::MySurfDescTest() {
}

MySurfDescTest::~MySurfDescTest() {
}

int MySurfDescTest::fRound(float flt) {
  return (int)floor(flt+0.5f);
}

//计算x方向的Harr小波响应值
float MySurfDescTest::haarX(Mat img, int row, int column, int s) {
    return BoxIntegral(img, row-s/2, column, s, s/2) 
        -1 * BoxIntegral(img, row-s/2, column-s/2, s, s/2);
}

//计算y方向的Harr小波响应值
float MySurfDescTest::haarY(Mat img, int row, int column, int s) {
    return BoxIntegral(img, row, column-s/2, s/2, s) 
        -1 * BoxIntegral(img, row-s/2, column-s/2, s/2, s);
}

//得到点(x,y)处相对于正x轴方向的角度
float MySurfDescTest::getAngle(float X, float Y) {
    if(X > 0 && Y >= 0) {
        return atan(Y/X);
    }
    if(X < 0 && Y >= 0) {
        return pi - atan(-Y/X);
    }
    if(X < 0 && Y < 0) {
        return pi + atan(Y/X);
    }
    if(X > 0 && Y < 0) {
        return 2*pi - atan(-Y/X);
    }
    return 0;
}

//计算(x,y)处的，方差为sig的2维高斯值
float MySurfDescTest::gaussian(int x, int y, float sig) {
  return (1.0f/(2.0f*pi*sig*sig)) * exp( -(x*x+y*y)/(2.0f*sig*sig));
}

void MySurfDescTest::GetOrientation(Mat integ_mat, MyKeyPoint &ipt) {
    float gauss = 0.f, scale = ipt.scale;
    const int s = fRound(scale), r = fRound(ipt.y), c = fRound(ipt.x);
    vector<float> resX(109), resY(109), Ang(109);

    //在半径为6倍尺度圆形区域内计算Ipoints的haar响应
    int idx = 0;
    for(int i = -6; i <= 6; ++i) {
        for(int j = -6; j <= 6; ++j) {
            if(i*i + j*j < 36) {
                gauss = gauss25[abs(i)][abs(j)]; 
                resX[idx] = gauss * haarX(integ_mat, r+j*s, c+i*s, 4*s);
                resY[idx] = gauss * haarY(integ_mat, r+j*s, c+i*s, 4*s);
                Ang[idx] = getAngle(resX[idx], resY[idx]);
                ++idx;
            }
        }
    }

    //计算主方向 
    float sumX=0.f, sumY=0.f;
    float max=0.f, orientation = 0.f;
    float ang1=0.f, ang2=0.f;

    //以特征点为中心，张角为pi/3的扇形滑动窗口
    for(ang1 = 0; ang1 < 2*pi;  ang1+=0.15f) {
        ang2 = (ang1+pi/3.0f > 2*pi ? ang1-5.0f*pi/3.0f : ang1+pi/3.0f);
        sumX = sumY = 0.f; 
        for(unsigned int k = 0; k < Ang.size(); ++k) {
            //确定关键点是否在窗口内
            if (ang1 < ang2 && ang1 < Ang[k] && Ang[k] < ang2) {
                sumX+=resX[k];  
                sumY+=resY[k];
            } else if (ang2 < ang1 && ((Ang[k] > 0 && Ang[k] < ang2) || (Ang[k] > ang1 && Ang[k] < 2*pi) )) {
                sumX+=resX[k];  
                sumY+=resY[k];
            }
        }

        //如果产生的矢量比所有的矢量都长,则将其方向作为新的主方向
        if (sumX*sumX + sumY*sumY > max) {
            //保存最长矢量的方向
            max = sumX*sumX + sumY*sumY;
            orientation = getAngle(sumX, sumY);
        }
    }

    //将主方向作为ipt的一个属性
    ipt.orientation = orientation;
}

//得到修正后的描述符
void MySurfDescTest::GetDescriptor(Mat integ_mat, MyKeyPoint &ipt) {
    int sample_x, sample_y, count=0;
    int i = 0, ix = 0, j = 0, jx = 0, xs = 0, ys = 0;
    float dx, dy, mdx, mdy;
    float gauss_s1 = 0.f, gauss_s2 = 0.f;
    float rx = 0.f, ry = 0.f, rrx = 0.f, rry = 0.f, len = 0.f;
    float cx = -0.5f, cy = 0.f; //以子窗口为中心取4x4高斯权重

    float scale = ipt.scale;
    int x = fRound(ipt.x);
    int y = fRound(ipt.y);  
    float co = cos(ipt.orientation);
    float si = sin(ipt.orientation);

    i = -8;

    //计算特征点的描述符
    while(i < 12) {
        j = -8;
        i = i-4;

        cx += 1.f;
        cy = -0.5f;

        while(j < 12) {
            dx=dy=mdx=mdy=0.f;
            cy += 1.f;

            j = j - 4;

            ix = i + 5;
            jx = j + 5;

            xs = fRound(x + ( -jx*scale*si + ix*scale*co));
            ys = fRound(y + ( jx*scale*co + ix*scale*si));

            for (int k = i; k < i + 9; ++k) {
                for (int l = j; l < j + 9; ++l) {
                    //得到样本点旋转后的坐标
                    sample_x = fRound(x + (-l*scale*si + k*scale*co));
                    sample_y = fRound(y + ( l*scale*co + k*scale*si));

                    //得到x和y的高斯加权响应值
                    gauss_s1 = gaussian(xs-sample_x,ys-sample_y,2.5f*scale);
                    rx = haarX(integ_mat, sample_y, sample_x, 2*fRound(scale));
                    ry = haarY(integ_mat, sample_y, sample_x, 2*fRound(scale));

                    //得到旋转后x和y的高斯加权响应值
                    rrx = gauss_s1*(-rx*si + ry*co);
                    rry = gauss_s1*(rx*co + ry*si);

                    dx += rrx;
                    dy += rry;
                    mdx += fabs(rrx);
                    mdy += fabs(rry);
                }
            }

            //将值添加到描述符向量中
            gauss_s2 = gaussian(cx-2.0f,cy-2.0f,1.5f);

            ipt.descriptor[count++] = dx*gauss_s2;
            ipt.descriptor[count++] = dy*gauss_s2;
            ipt.descriptor[count++] = mdx*gauss_s2;
            ipt.descriptor[count++] = mdy*gauss_s2;

            len += (dx*dx + dy*dy + mdx*mdx + mdy*mdy) * gauss_s2*gauss_s2;

            j += 9;
        }
        i += 9;
    }

    //转换为单位向量
    len = sqrt(len);
    for(int i = 0; i < 64; ++i) {
        ipt.descriptor[i] /= len;
    }
}

//画出所有特征点向量
Mat MySurfDescTest::DispDesc(Mat src, vector<MyKeyPoint> key_point_vec) {
    Mat img = src.clone();
    float s, o;
    int r1, c1, r2, c2, lap;

    for(unsigned int i = 0; i<key_point_vec.size(); i++) {
        MyKeyPoint ipt = key_point_vec[i];
        s = (2.5f * ipt.scale);
        o = ipt.orientation;
        lap = ipt.laplacian;
        r1 = fRound(ipt.y);
        c1 = fRound(ipt.x);
        c2 = fRound(s*cos(o)) + c1;
        r2 = fRound(s*sin(o)) + r1;

        //绿线表示方向
        line(img, cvPoint(c1, r1), cvPoint(c2, r2), cvScalar(0, 255, 0));

        if (lap == 1) { //蓝色圆圈表示亮背景上的暗斑
            circle(img, cvPoint(c1,r1), fRound(s), cvScalar(255, 0, 0),1);
        } else if (lap == 0) { //红色圆圈表示暗背景上的亮斑
            circle(img, cvPoint(c1,r1), fRound(s), cvScalar(0, 0, 255),1);
        }
    }
    return img;
}

void MySurfDescTest::run(Mat integ_mat, vector<MyKeyPoint> &key_point_vec) {
    for(int i=0; i<key_point_vec.size(); i++) {
        GetOrientation(integ_mat, key_point_vec[i]);
        GetDescriptor(integ_mat, key_point_vec[i]);
    }
}
