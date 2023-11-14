#include "ic_ecc.hpp"

MyIcEccTest::MyIcEccTest() {
}

MyIcEccTest::~MyIcEccTest() {
}

void MyIcEccTest::image_jacobian_translation_ECC(const Mat& src1, const Mat& src2, Mat& dst) {
    const int w = src1.cols;

    //compute Jacobian blocks (2 blocks)
    src1.copyTo(dst.colRange(0, w));
    src2.copyTo(dst.colRange(w, 2*w));
}

void MyIcEccTest::project_onto_jacobian_ECC(const Mat& src1, const Mat& src2, Mat& dst){
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


void MyIcEccTest::update_warping_matrix_ECC (Mat& map_matrix, const Mat& update, const int motionType) {
    float* mapPtr = map_matrix.ptr<float>(0);
    const float* updatePtr = update.ptr<float>(0);

    mapPtr[2] += updatePtr[0];
    mapPtr[5] += updatePtr[1];
}

double MyIcEccTest::findTransformECC(Mat src, Mat dst, Mat &map, int motionType,
                int number_of_iterations, float termination_eps, int gaussFiltSize) {
    map.create(2, 3, CV_32FC1);
    map = Mat::eye(2, 3, CV_32F);

    const int numberOfParameters = 2;

    const int ws = src.cols;
    const int hs = src.rows;
    const int wd = dst.cols;
    const int hd = dst.rows;

    Mat templateZM    = Mat(hs, ws, CV_32F);// to store the (smoothed)zero-mean version of template
    Mat templateFloat = Mat(hs, ws, CV_32F);// to store the (smoothed) template
    Mat imageFloat    = Mat(hd, wd, CV_32F);// to store the (smoothed) input image
    Mat imageWarped   = Mat(hs, ws, CV_32F);// to store the warped zero-mean input image

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
    filter2D(templateFloat, gradientX, -1, dx);
    filter2D(templateFloat, gradientY, -1, dx.t());

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
    
    image_jacobian_translation_ECC(gradientX, gradientY, jacobian);
    // calculate Hessian and its inverse
    project_onto_jacobian_ECC(jacobian, jacobian, hessian);
    hessianInv = hessian.inv();
    
    Scalar imgMean, imgStd, tmpMean, tmpStd;
    meanStdDev(templateFloat, tmpMean, tmpStd);
    subtract(templateFloat, tmpMean, templateZM);//zero-mean template
    const double tmpNorm = std::sqrt(hd*wd*(tmpStd.val[0])*(tmpStd.val[0]));
    project_onto_jacobian_ECC(jacobian, templateZM, templateProjection);


    // iteratively update map_matrix
    const int imageFlags = INTER_LINEAR  + WARP_INVERSE_MAP;
    double rho      = -1;
    double last_rho = - termination_eps;
    for (int i = 1; (i <= number_of_iterations) && (fabs(rho-last_rho)>= termination_eps); i++) {
        warpAffine(imageFloat, imageWarped,     map, imageWarped.size(),     imageFlags);

        meanStdDev(imageWarped,   imgMean, imgStd);
        subtract(imageWarped,   imgMean, imageWarped);//zero-mean input

        const double imgNorm = std::sqrt(hd*wd*(imgStd.val[0])*(imgStd.val[0]));
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

        // calculate the parameter lambda to account for illumination variation
        imageProjectionHessian = hessianInv*imageProjection;
        const double lambda_n = (imgNorm*imgNorm) - imageProjection.dot(imageProjectionHessian);
        const double lambda_d = correlation - templateProjection.dot(imageProjectionHessian);
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
