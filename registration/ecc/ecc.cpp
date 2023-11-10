#include "ecc.hpp"

MyEccTest::MyEccTest() {
}

MyEccTest::~MyEccTest() {
}


void MyEccTest::image_jacobian_homo_ECC(const Mat& src1, const Mat& src2,
                                    const Mat& src3, const Mat& src4,
                                    const Mat& src5, Mat& dst) {
    const float* hptr = src5.ptr<float>(0);
    const float h0_ = hptr[0];
    const float h1_ = hptr[3];
    const float h2_ = hptr[6];
    const float h3_ = hptr[1];
    const float h4_ = hptr[4];
    const float h5_ = hptr[7];
    const float h6_ = hptr[2];
    const float h7_ = hptr[5];

    const int w = src1.cols;

    //create denominator for all points as a block
    Mat den_ = src3*h2_ + src4*h5_ + 1.0;//check the time of this! otherwise use addWeighted

    //create projected points
    Mat hatX_ = -src3*h0_ - src4*h3_ - h6_;
    divide(hatX_, den_, hatX_);
    Mat hatY_ = -src3*h1_ - src4*h4_ - h7_;
    divide(hatY_, den_, hatY_);

    //instead of dividing each block with den,
    //just pre-divide the block of gradients (it's more efficient)

    Mat src1Divided_;
    Mat src2Divided_;

    divide(src1, den_, src1Divided_);
    divide(src2, den_, src2Divided_);


    //compute Jacobian blocks (8 blocks)
    dst.colRange(0, w) = src1Divided_.mul(src3);//1
    dst.colRange(w,2*w) = src2Divided_.mul(src3);//2

    Mat temp_ = (hatX_.mul(src1Divided_)+hatY_.mul(src2Divided_));
    dst.colRange(2*w,3*w) = temp_.mul(src3);//3

    hatX_.release();
    hatY_.release();

    dst.colRange(3*w, 4*w) = src1Divided_.mul(src4);//4
    dst.colRange(4*w, 5*w) = src2Divided_.mul(src4);//5
    dst.colRange(5*w, 6*w) = temp_.mul(src4);//6
    src1Divided_.copyTo(dst.colRange(6*w, 7*w));//7
    src2Divided_.copyTo(dst.colRange(7*w, 8*w));//8
}

void MyEccTest::image_jacobian_euclidean_ECC(const Mat& src1, const Mat& src2,
                                         const Mat& src3, const Mat& src4,
                                         const Mat& src5, Mat& dst) {

    CV_Assert( src1.size()==src2.size());
    CV_Assert( src1.size()==src3.size());
    CV_Assert( src1.size()==src4.size());

    CV_Assert( src1.rows == dst.rows);
    CV_Assert(dst.cols == (src1.cols*3));
    CV_Assert(dst.type() == CV_32FC1);

    CV_Assert(src5.isContinuous());

    const float* hptr = src5.ptr<float>(0);

    const float h0 = hptr[0];//cos(theta)
    const float h1 = hptr[3];//sin(theta)

    const int w = src1.cols;

    //create -sin(theta)*X -cos(theta)*Y for all points as a block -> hatX
    Mat hatX = -(src3*h1) - (src4*h0);

    //create cos(theta)*X -sin(theta)*Y for all points as a block -> hatY
    Mat hatY = (src3*h0) - (src4*h1);

    //compute Jacobian blocks (3 blocks)
    dst.colRange(0, w) = (src1.mul(hatX))+(src2.mul(hatY));//1

    src1.copyTo(dst.colRange(w, 2*w));//2
    src2.copyTo(dst.colRange(2*w, 3*w));//3
}


void MyEccTest::image_jacobian_affine_ECC(const Mat& src1, const Mat& src2,
                                      const Mat& src3, const Mat& src4,
                                      Mat& dst) {
    const int w = src1.cols;

    //compute Jacobian blocks (6 blocks)
    dst.colRange(0,w) = src1.mul(src3);//1
    dst.colRange(w,2*w) = src2.mul(src3);//2
    dst.colRange(2*w,3*w) = src1.mul(src4);//3
    dst.colRange(3*w,4*w) = src2.mul(src4);//4
    src1.copyTo(dst.colRange(4*w,5*w));//5
    src2.copyTo(dst.colRange(5*w,6*w));//6
}

void MyEccTest::image_jacobian_translation_ECC(const Mat& src1, const Mat& src2, Mat& dst) {
    const int w = src1.cols;

    //compute Jacobian blocks (2 blocks)
    src1.copyTo(dst.colRange(0, w));
    src2.copyTo(dst.colRange(w, 2*w));
}

void MyEccTest::project_onto_jacobian_ECC(const Mat& src1, const Mat& src2, Mat& dst){
    /* this functions is used for two types of projections. If src1.cols ==src.cols
    it does a blockwise multiplication (like in the outer product of vectors)
    of the blocks in matrices src1 and src2 and dst
    has size (number_of_blcks x number_of_blocks), otherwise dst is a vector of size
    (number_of_blocks x 1) since src2 is "multiplied"(dot) with each block of src1.

    The number_of_blocks is equal to the number of parameters we are lloking for
    (i.e. rtanslation:2, euclidean: 3, affine: 6, homography: 8)

    */
    int w;

    float* dstPtr = dst.ptr<float>(0);

    if (src1.cols !=src2.cols){//dst.cols==1
        w  = src2.cols;
        for (int i=0; i<dst.rows; i++){
            dstPtr[i] = (float) src2.dot(src1.colRange(i*w,(i+1)*w));
        }
    } else {
        w = src2.cols/dst.cols;
        Mat mat;
        for (int i=0; i<dst.rows; i++){
            mat = Mat(src1.colRange(i*w, (i+1)*w));
            dstPtr[i*(dst.rows+1)] = (float) pow(norm(mat),2); //diagonal elements

            for (int j=i+1; j<dst.cols; j++){ //j starts from i+1
                dstPtr[i*dst.cols+j] = (float) mat.dot(src2.colRange(j*w, (j+1)*w));
                dstPtr[j*dst.cols+i] = dstPtr[i*dst.cols+j]; //due to symmetry
            }
        }
    }
}


void MyEccTest::update_warping_matrix_ECC (Mat& map_matrix, const Mat& update, const int motionType) {
    if (motionType == MOTION_HOMOGRAPHY)
        CV_Assert (map_matrix.rows == 3 && update.rows == 8);
    else if (motionType == MOTION_AFFINE)
        CV_Assert(map_matrix.rows == 2 && update.rows == 6);
    else if (motionType == MOTION_EUCLIDEAN)
        CV_Assert (map_matrix.rows == 2 && update.rows == 3);
    else
        CV_Assert (map_matrix.rows == 2 && update.rows == 2);

    float* mapPtr = map_matrix.ptr<float>(0);
    const float* updatePtr = update.ptr<float>(0);

    if (motionType == MOTION_TRANSLATION){
        mapPtr[2] += updatePtr[0];
        mapPtr[5] += updatePtr[1];
    }
    if (motionType == MOTION_AFFINE) {
        mapPtr[0] += updatePtr[0];
        mapPtr[3] += updatePtr[1];
        mapPtr[1] += updatePtr[2];
        mapPtr[4] += updatePtr[3];
        mapPtr[2] += updatePtr[4];
        mapPtr[5] += updatePtr[5];
    }
    if (motionType == MOTION_HOMOGRAPHY) {
        mapPtr[0] += updatePtr[0];
        mapPtr[3] += updatePtr[1];
        mapPtr[6] += updatePtr[2];
        mapPtr[1] += updatePtr[3];
        mapPtr[4] += updatePtr[4];
        mapPtr[7] += updatePtr[5];
        mapPtr[2] += updatePtr[6];
        mapPtr[5] += updatePtr[7];
    }
    if (motionType == MOTION_EUCLIDEAN) {
        double new_theta = updatePtr[0];
        new_theta += asin(mapPtr[3]);

        mapPtr[2] += updatePtr[1];
        mapPtr[5] += updatePtr[2];
        mapPtr[0] = mapPtr[4] = (float) cos(new_theta);
        mapPtr[3] = (float) sin(new_theta);
        mapPtr[1] = -mapPtr[3];
    }
}

double MyEccTest::findTransformECC(Mat src, Mat dst, Mat &map, int motionType,
                int number_of_iterations, float termination_eps, int gaussFiltSize) {
    // If the user passed an un-initialized warpMatrix, initialize to identity
    if(map.empty()) {
        int rowCount = 2;
        if(motionType == MOTION_HOMOGRAPHY)
            rowCount = 3;

        map.create(rowCount, 3, CV_32FC1);
        map = Mat::eye(rowCount, 3, CV_32F);
    }

    int paramTemp = 6;//default: affine
    switch (motionType){
      case MOTION_TRANSLATION:
          paramTemp = 2;
          break;
      case MOTION_EUCLIDEAN:
          paramTemp = 3;
          break;
      case MOTION_HOMOGRAPHY:
          paramTemp = 8;
          break;
    }

    const int numberOfParameters = paramTemp;

    const int ws = src.cols;
    const int hs = src.rows;
    const int wd = dst.cols;
    const int hd = dst.rows;

    Mat Xcoord = Mat(1, ws, CV_32F);
    Mat Ycoord = Mat(hs, 1, CV_32F);
    Mat Xgrid = Mat(hs, ws, CV_32F);
    Mat Ygrid = Mat(hs, ws, CV_32F);

    float* XcoPtr = Xcoord.ptr<float>(0);
    float* YcoPtr = Ycoord.ptr<float>(0);
    int j;
    for (j=0; j<ws; j++)
        XcoPtr[j] = (float) j;
    for (j=0; j<hs; j++)
        YcoPtr[j] = (float) j;

    repeat(Xcoord, hs, 1, Xgrid);
    repeat(Ycoord, 1, ws, Ygrid);

    Xcoord.release();
    Ycoord.release();

    Mat templateZM    = Mat(hs, ws, CV_32F);// to store the (smoothed)zero-mean version of template
    Mat templateFloat = Mat(hs, ws, CV_32F);// to store the (smoothed) template
    Mat imageFloat    = Mat(hd, wd, CV_32F);// to store the (smoothed) input image
    Mat imageWarped   = Mat(hs, ws, CV_32F);// to store the warped zero-mean input image
    Mat imageMask     = Mat(hs, ws, CV_8U); // to store the final mask

    Mat preMask = Mat::ones(hd, wd, CV_8U);

    //gaussian filtering is optional
    src.convertTo(templateFloat, templateFloat.type());
    GaussianBlur(templateFloat, templateFloat, Size(gaussFiltSize, gaussFiltSize), 0, 0);

    dst.convertTo(imageFloat, imageFloat.type());
    GaussianBlur(imageFloat, imageFloat, Size(gaussFiltSize, gaussFiltSize), 0, 0);

    // needed matrices for gradients and warped gradients
    Mat gradientX = Mat::zeros(hd, wd, CV_32FC1);
    Mat gradientY = Mat::zeros(hd, wd, CV_32FC1);
    Mat gradientXWarped = Mat(hs, ws, CV_32FC1);
    Mat gradientYWarped = Mat(hs, ws, CV_32FC1);

    // calculate first order image derivatives
    Matx13f dx(-0.5f, 0.0f, 0.5f);

    filter2D(imageFloat, gradientX, -1, dx);
    filter2D(imageFloat, gradientY, -1, dx.t());

    // matrices needed for solving linear equation system for maximizing ECC
    Mat jacobian                = Mat(hs, ws*numberOfParameters, CV_32F);
    Mat hessian                 = Mat(numberOfParameters, numberOfParameters, CV_32F);
    Mat hessianInv              = Mat(numberOfParameters, numberOfParameters, CV_32F);
    Mat imageProjection         = Mat(numberOfParameters, 1, CV_32F);
    Mat templateProjection      = Mat(numberOfParameters, 1, CV_32F);
    Mat imageProjectionHessian  = Mat(numberOfParameters, 1, CV_32F);
    Mat errorProjection         = Mat(numberOfParameters, 1, CV_32F);

    Mat deltaP = Mat(numberOfParameters, 1, CV_32F);//transformation parameter correction
    Mat error = Mat(hs, ws, CV_32F);//error as 2D matrix

    const int imageFlags = INTER_LINEAR  + WARP_INVERSE_MAP;
    const int maskFlags  = INTER_NEAREST + WARP_INVERSE_MAP;

    // iteratively update map_matrix
    double rho      = -1;
    double last_rho = - termination_eps;
    for (int i = 1; (i <= number_of_iterations) && (fabs(rho-last_rho)>= termination_eps); i++) {

        // warp-back portion of the inputImage and gradients to the coordinate space of the templateImage
        if (motionType != MOTION_HOMOGRAPHY) {
            warpAffine(imageFloat, imageWarped,     map, imageWarped.size(),     imageFlags);
            warpAffine(gradientX,  gradientXWarped, map, gradientXWarped.size(), imageFlags);
            warpAffine(gradientY,  gradientYWarped, map, gradientYWarped.size(), imageFlags);
        } else {
            warpPerspective(imageFloat, imageWarped,     map, imageWarped.size(),     imageFlags);
            warpPerspective(gradientX,  gradientXWarped, map, gradientXWarped.size(), imageFlags);
            warpPerspective(gradientY,  gradientYWarped, map, gradientYWarped.size(), imageFlags);
        }

        Scalar imgMean, imgStd, tmpMean, tmpStd;
        meanStdDev(imageWarped,   imgMean, imgStd);
        meanStdDev(templateFloat, tmpMean, tmpStd);

        subtract(imageWarped,   imgMean, imageWarped);//zero-mean input
        templateZM = Mat::zeros(templateZM.rows, templateZM.cols, templateZM.type());
        subtract(templateFloat, tmpMean, templateZM);//zero-mean template

        const double tmpNorm = std::sqrt(hd*wd*(tmpStd.val[0])*(tmpStd.val[0]));
        const double imgNorm = std::sqrt(hd*wd*(imgStd.val[0])*(imgStd.val[0]));

        // calculate jacobian of image wrt parameters
        switch (motionType){
            case MOTION_AFFINE:
                image_jacobian_affine_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, jacobian);
                break;
            case MOTION_HOMOGRAPHY:
                image_jacobian_homo_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, map, jacobian);
                break;
            case MOTION_TRANSLATION:
                image_jacobian_translation_ECC(gradientXWarped, gradientYWarped, jacobian);
                break;
            case MOTION_EUCLIDEAN:
                image_jacobian_euclidean_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, map, jacobian);
                break;
        }

        // calculate Hessian and its inverse
        project_onto_jacobian_ECC(jacobian, jacobian, hessian);

        hessianInv = hessian.inv();

        const double correlation = templateZM.dot(imageWarped);

        // calculate enhanced correlation coefficient (ECC)->rho
        last_rho = rho;
        rho = correlation/(imgNorm*tmpNorm);

        cout << "!!!!!! rho:" << rho << ", i:" << i << endl;
        if (cvIsNaN(rho)) {
          CV_Error(Error::StsNoConv, "NaN encountered.");
        }

        // project images into jacobian
        project_onto_jacobian_ECC( jacobian, imageWarped, imageProjection);
        project_onto_jacobian_ECC(jacobian, templateZM, templateProjection);

        // calculate the parameter lambda to account for illumination variation
        imageProjectionHessian = hessianInv*imageProjection;
        const double lambda_n = (imgNorm*imgNorm) - imageProjection.dot(imageProjectionHessian);
        const double lambda_d = correlation - templateProjection.dot(imageProjectionHessian);
        if (lambda_d <= 0.0) {
            rho = -1;
            CV_Error(Error::StsNoConv, "The algorithm stopped before its convergence. The correlation is going to be minimized. Images may be uncorrelated or non-overlapped");

        }
        const double lambda = (lambda_n/lambda_d);

        // estimate the update step delta_p
        error = lambda*templateZM - imageWarped;
        project_onto_jacobian_ECC(jacobian, error, errorProjection);
        deltaP = hessianInv * errorProjection;

        // update warping matrix
        update_warping_matrix_ECC( map, deltaP, motionType);
    }

    // return final correlation coefficient
    return rho;
}
