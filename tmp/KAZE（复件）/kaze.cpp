
//=============================================================================
//
// KAZE.cpp
// Author: Pablo F. Alcantarilla
// Institution: University d'Auvergne
// Address: Clermont Ferrand, France
// Date: 21/01/2012
// Email: pablofdezalc@gmail.com
//
// KAZE Features Copyright 2012, Pablo F. Alcantarilla
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file KAZE.cpp
 * @brief Main class for detecting and describing features in a nonlinear
 * scale space
 * @date Jan 21, 2012
 * @author Pablo F. Alcantarilla
 * @update 2013-03-28 by Yuhua Zou
 *         Code optimization has been implemented via using 
 *         OpenMP and Boost Thread for multi-threading,
 *         changing the way to access matrix elements, etc.
 */

#include "KAZE.h"
#include "kaze_config.h"
#include <iostream>
#include <functional>

#if HAVE_BOOST_THREADING
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/bind.hpp>
#endif

// Namespaces
using namespace std;

/**
 * @brief KAZE default constructor
 * @note The constructor does not allocate memory for the nonlinear scale space
 */
//KAZE::KAZE(void)
//{
//    soffset = DEFAULT_SCALE_OFFSET;
//    sderivatives = DEFAULT_SIGMA_SMOOTHING_DERIVATIVES;
//    omax = DEFAULT_OCTAVE_MAX;
//    nsublevels = DEFAULT_NSUBLEVELS;
//  save_scale_space = DEFAULT_SAVE_SCALE_SPACE;
//    verbosity = DEFAULT_VERBOSITY;
//    kcontrast = DEFAULT_KCONTRAST;
//    descriptor_mode = DEFAULT_DESCRIPTOR_MODE;
//    use_upright = DEFAULT_UPRIGHT;
//  use_extended = DEFAULT_EXTENDED;
//  diffusivity = DEFAULT_DIFFUSIVITY_TYPE;
//    tkcontrast = 0.0;
//    tnlscale = 0.0;
//    tdetector = 0.0;
//    tmderivatives = 0.0;
//    tdescriptor = 0.0;
//    tsubpixel = 0.0;
//    img_width = 0;
//    img_height = 0;
//}

//*******************************************************************************
//*******************************************************************************

/**
 * @brief KAZE constructor with input options
 * @param options KAZE configuration options
 * @note The constructor allocates memory for the nonlinear scale space
*/
KAZE::KAZE(toptions &options)
{
    img_width = options.img_width;
    img_height = options.img_height;
    soffset = options.soffset;
    sderivatives = options.sderivatives;
    omax = options.omax;
    nsublevels = options.nsublevels;
    save_scale_space = options.save_scale_space;
    verbosity = options.verbosity;
    dthreshold = options.dthreshold;
    diffusivity = options.diffusivity;
    descriptor_mode = options.descriptor;
    use_upright = options.upright;
    use_extended = options.extended;

    kcontrast = DEFAULT_KCONTRAST;
    tkcontrast = 0.0;
    tnlscale = 0.0;
    tdetector = 0.0;
    tmderivatives = 0.0;
    tdresponse = 0.0;
    tdescriptor = 0.0;

    omax = options.omax > 0 ? options.omax : cvRound(std::log( (double)std::min( options.img_height, options.img_width ) ) / std::log(2.) - 2);
    //dthreshold = DEFAULT_DETECTOR_THRESHOLD + floorf( img_width/256.0f ) * 0.0015;

    // Now allocate memory for the evolution
    Allocate_Memory_Evolution();
}

//*******************************************************************************
//*******************************************************************************

/**
 * @brief This method allocates the memory for the nonlinear diffusion evolution
*/
void KAZE::Allocate_Memory_Evolution(void)
{
    // Allocate the dimension of the matrices for the evolution
    for( int i = 0; i <= omax-1; i++ )
    {        
        for( int j = 0; j <= nsublevels-1; j++ )
        {
             tevolution aux;
             aux.Lx  = cv::Mat::zeros(img_height,img_width,CV_32F);
             aux.Ly  = cv::Mat::zeros(img_height,img_width,CV_32F);
    
             aux.Lxx = cv::Mat::zeros(img_height,img_width,CV_32F);
             aux.Lxy = cv::Mat::zeros(img_height,img_width,CV_32F);
             aux.Lyy = cv::Mat::zeros(img_height,img_width,CV_32F);
             aux.Lflow = cv::Mat::zeros(img_height,img_width,CV_32F);
             aux.Lt  = cv::Mat::zeros(img_height,img_width,CV_32F);
             aux.Lsmooth = cv::Mat::zeros(img_height,img_width,CV_32F);
             aux.Lstep = cv::Mat::zeros(img_height,img_width,CV_32F);
             aux.Ldet = cv::Mat::zeros(img_height,img_width,CV_32F);

             aux.esigma = soffset*pow((float)2.0,(float)(j)/(float)(nsublevels) + i);
             aux.etime = 0.5*(aux.esigma*aux.esigma);
             aux.sigma_size = fRound(aux.esigma);

             aux.octave = i;
             aux.sublevel = j;
             evolution.push_back(aux);
        }
    }    
    
    // Allocate memory for the auxiliary variables that are used in the AOS scheme
    Ltx = cv::Mat::zeros(img_width,img_height,CV_32F);
    Lty = cv::Mat::zeros(img_height,img_width,CV_32F);
    px = cv::Mat::zeros(img_height,img_width,CV_32F);
    py = cv::Mat::zeros(img_height,img_width,CV_32F);
    ax = cv::Mat::zeros(img_height,img_width,CV_32F);
    ay = cv::Mat::zeros(img_height,img_width,CV_32F);
    bx = cv::Mat::zeros(img_height-1,img_width,CV_32F);
    by = cv::Mat::zeros(img_height-1,img_width,CV_32F);
    qr = cv::Mat::zeros(img_height-1,img_width,CV_32F);
    qc = cv::Mat::zeros(img_height,img_width-1,CV_32F);
    
}

//*******************************************************************************
//*******************************************************************************

/**
 * @brief This method creates the nonlinear scale space for a given image
 * @param img Input image for which the nonlinear scale space needs to be created
 * @return 0 if the nonlinear scale space was created successfully. -1 otherwise
*/
int KAZE::Create_Nonlinear_Scale_Space(const cv::Mat &img)
{
    if( verbosity == true )
    {
        std::cout << "\n> Creating nonlinear scale space." << std::endl;
    }

    double t2 = 0.0, t1 = 0.0;

    if( evolution.size() == 0 )
    {
        std::cout << "---> Error generating the nonlinear scale space!!" << std::endl;
        std::cout << "---> Firstly you need to call KAZE::Allocate_Memory_Evolution()" << std::endl;
        return -1;
    }

    int64 start_t1 = cv::getTickCount();

    // Copy the original image to the first level of the evolution
    if( verbosity == true )
    {
        std::cout << "-> Perform the Gaussian smoothing." << std::endl;
    }

    img.copyTo(evolution[0].Lt);
    Gaussian_2D_Convolution(evolution[0].Lt,evolution[0].Lt,0,0,soffset);
    Gaussian_2D_Convolution(evolution[0].Lt,evolution[0].Lsmooth,0,0,sderivatives);

    // Firstly compute the kcontrast factor
    Compute_KContrast(evolution[0].Lt,KCONTRAST_PERCENTILE);
    
    t2 = cv::getTickCount();
    tkcontrast = 1000.0 * (t2 - start_t1) / cv::getTickFrequency();
    
    if( verbosity == true )
    {
        std::cout << "-> Computed K-contrast factor. Execution time (ms): " << tkcontrast << std::endl;
        std::cout << "-> Now computing the nonlinear scale space!!" << std::endl;
    }

    // Now generate the rest of evolution levels
    for( unsigned int i = 1; i < evolution.size(); i++ )
    {
        Gaussian_2D_Convolution(evolution[i-1].Lt,evolution[i].Lsmooth,0,0,sderivatives);

        // Compute the Gaussian derivatives Lx and Ly
        Image_Derivatives_Scharr(evolution[i].Lsmooth,evolution[i].Lx,1,0);
        Image_Derivatives_Scharr(evolution[i].Lsmooth,evolution[i].Ly,0,1);

        // Compute the conductivity equation
        if( diffusivity == 0 )
        {
            PM_G1(evolution[i].Lsmooth,evolution[i].Lflow,evolution[i].Lx,evolution[i].Ly,kcontrast);
        }
        else if( diffusivity == 1 )
        {
            PM_G2(evolution[i].Lsmooth,evolution[i].Lflow,evolution[i].Lx,evolution[i].Ly,kcontrast);
        }
        else if( diffusivity == 2 )
        {
            Weickert_Diffusivity(evolution[i].Lsmooth,evolution[i].Lflow,evolution[i].Lx,evolution[i].Ly,kcontrast);
        }
    
        // Perform the evolution step with AOS
#if HAVE_BOOST_THREADING
        AOS_Step_Scalar_Parallel(evolution[i].Lt,evolution[i-1].Lt,evolution[i].Lflow,evolution[i].etime-evolution[i-1].etime);
#else
        AOS_Step_Scalar(evolution[i].Lt,evolution[i-1].Lt,evolution[i].Lflow,evolution[i].etime-evolution[i-1].etime);
#endif

        if( verbosity == true )
        {
            std::cout << "--> Computed image evolution step " << i << " Evolution time: " << evolution[i].etime << 
            " Sigma: " << evolution[i].esigma << std::endl;
        }
    }

    
    t2 = cv::getTickCount();
    tnlscale = 1000.0*(t2-start_t1) / cv::getTickFrequency();

    if( verbosity == true )
    {
        std::cout << "> Computed the nonlinear scale space. Execution time (ms): " << tnlscale << std::endl;
    }

    return 0;        
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the k contrast factor
 * @param img Input image
 * @param kpercentile Percentile of the gradient histogram
*/
void KAZE::Compute_KContrast(const cv::Mat &img, const float &kpercentile)
{
    if( verbosity == true )
    {
        std::cout << "-> Computing Kcontrast factor." << std::endl;
    }

    if( COMPUTE_KCONTRAST == true )
    {
        kcontrast = Compute_K_Percentile(img,kpercentile,sderivatives,KCONTRAST_NBINS,0,0);
    }
    
    if( verbosity == true )
    {
        std::cout << "--> kcontrast = " << kcontrast << std::endl;
    }    
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the multiscale derivatives for the nonlinear scale space
*/
void KAZE::Compute_Multiscale_Derivatives(void)
{
    int64 t1 = cv::getTickCount();

    int N = img_width * img_height;

    #pragma omp parallel for
    for( int i = 0; i < evolution.size(); i++ )
    {
        if( verbosity == true )
        {
            std::cout << "--> Multiscale derivatives. Sigma ("<< i <<"): " << evolution[i].sigma_size << ". Thread: " << omp_get_thread_num() << std::endl;
        }
        
        // Compute multiscale derivatives for the detector
        Compute_Scharr_Derivatives(evolution[i].Lsmooth,evolution[i].Lx,1,0,evolution[i].sigma_size);
        Compute_Scharr_Derivatives(evolution[i].Lsmooth,evolution[i].Ly,0,1,evolution[i].sigma_size);
        Compute_Scharr_Derivatives(evolution[i].Lx,evolution[i].Lxx,1,0,evolution[i].sigma_size);
        Compute_Scharr_Derivatives(evolution[i].Ly,evolution[i].Lyy,0,1,evolution[i].sigma_size);
        Compute_Scharr_Derivatives(evolution[i].Lx,evolution[i].Lxy,0,1,evolution[i].sigma_size);

        int esigma = evolution[i].sigma_size, esigma2 = esigma*esigma;
        for ( int j = 0; j < N; j++ )
        {
            *( evolution[i].Lx.ptr<float>(0)+j ) *= esigma;
            *( evolution[i].Ly.ptr<float>(0)+j ) *= esigma;
            *( evolution[i].Lxx.ptr<float>(0)+j ) *= esigma2;
            *( evolution[i].Lxy.ptr<float>(0)+j ) *= esigma2;
            *( evolution[i].Lyy.ptr<float>(0)+j ) *= esigma2;
        }
    }
    
    int64 t2 = cv::getTickCount();
    tmderivatives = 1000.0 * (t2-t1) / cv::getTickFrequency();

}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the feature detector response for the nonlinear scale space
 * @note We use the Hessian determinant as feature detector
*/
void KAZE::Compute_Detector_Response(void)
{
    float lxx = 0.0, lxy = 0.0, lyy = 0.0;
    float *ptr;

    int64 t1 = cv::getTickCount(), t2 = 0;

    // Firstly compute the multiscale derivatives
    Compute_Multiscale_Derivatives();

    t2 = cv::getTickCount();
    tdresponse = 1000.0 * (t2-t1) / cv::getTickFrequency();
    if( verbosity == true )
    {
        std::cout << "-> Computed multiscale derivatives. Execution time (ms): " << tdresponse << std::endl;
    }
    t1 = cv::getTickCount();

    int N = img_width * img_height;
    for( int i = 0; i < evolution.size(); i++ )
    {  
        for( int jx = 0; jx < N; jx++ )
        {
            // Get values of lxx,lxy,and lyy
            ptr = evolution[i].Lxx.ptr<float>(0);
            lxx = ptr[jx];

            ptr = evolution[i].Lxy.ptr<float>(0);
            lxy = ptr[jx];

            ptr = evolution[i].Lyy.ptr<float>(0);
            lyy = ptr[jx];

            // Compute ldet
            ptr = evolution[i].Ldet.ptr<float>(0);
            ptr[jx] = (lxx*lyy-lxy*lxy);
        }
    }
    
    t2 = cv::getTickCount();
    tdresponse = 1000.0 * (t2-t1) / cv::getTickFrequency();
    if( verbosity == true )
    {
        std::cout << "-> Computed Hessian determinant. Execution time (ms): " << tdresponse << std::endl;
    }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method selects interesting keypoints through the nonlinear scale space
*/
void KAZE::Feature_Detection(std::vector<Ipoint> &kpts)
{    
    if( verbosity == true )
    {
        std::cout << "\n> Detecting features. " << std::endl;
    }        
    int64 t1 = cv::getTickCount(), t2 = 0;

    // Firstly compute the detector response for each pixel and scale level
    Compute_Detector_Response();

    t2 = cv::getTickCount();
    double tresponse  = 1000.0 * (t2-t1) / cv::getTickFrequency();
    if( verbosity == true )
    {
        std::cout << "-> Computed detector response. Execution time (ms):" << tresponse << std::endl;
    }
    int64 t13 = cv::getTickCount();    

    // Find scale space extrema
    Determinant_Hessian_Parallel(kpts);

    t2 = cv::getTickCount();
    double thessian  = 1000.0 * (t2-t13) / cv::getTickFrequency();
    if( verbosity == true )
    {
        std::cout << "-> Computed Hessian determinant. Execution time (ms):" << thessian << std::endl;
    }    

    // Perform some subpixel refinement
    if( SUBPIXEL_REFINEMENT == true )
    {
        Do_Subpixel_Refinement(kpts);
    }

    t2 = cv::getTickCount();
    tdetector = 1000.0*(t2-t1) / cv::getTickFrequency();
    if( verbosity == true )
    {
        std::cout << "> Feature detection done. Execution time (ms): " << tdetector << std::endl;
    }

}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method performs the detection of keypoints by using the normalized
 * score of the Hessian determinant through the nonlinear scale space
 * @note We compute features for each of the nonlinear scale space level in a different processing thread
*/
void KAZE::Determinant_Hessian_Parallel(std::vector<Ipoint> &kpts)
{
    unsigned int level = 0;
    float dist = 0.0, smax = 3.0;
    int npoints = 0, id_repeated = 0;
    int left_x = 0, right_x = 0, up_y = 0, down_y = 0;
    bool is_extremum = false, is_repeated = false, is_out = false;
    int64 t1 = cv::getTickCount(), t2 = 0;

    // Delete the memory of the vector of keypoints vectors
    // In case we use the same kaze object for multiple images
    vector<vector<Ipoint> >(evolution.size()-2, vector<Ipoint>()).swap(kpts_par);

    t2 = cv::getTickCount();
    if( verbosity == true )
    {
        std::cout << "--> Init kpts_par time: "<< 1000.0*(t2-t1)/cv::getTickFrequency() << std::endl;
    }    
    t1 = cv::getTickCount();

    // Find extremum at each scale level
#if HAVE_BOOST_THREADING
    // Create multi-thread
    boost::thread_group mthreads;

    for( unsigned int i = 1; i < evolution.size()-1; i++ )
    {    
        // Create the thread for finding extremum at i scale level
        mthreads.create_thread(boost::bind(&KAZE::Find_Extremum_Threading,this,i));
    }

    // Wait for the threads
    mthreads.join_all();
#else
    #pragma omp parallel for
    for( int n = 1; n < evolution.size()-1; n++ )
    {    
        Find_Extremum_Threading(n);
    }
#endif    

    t2 = cv::getTickCount();
    if( verbosity == true )
    {
        std::cout << "--> Find extremum time: "<< 1000.0*(t2-t1)/cv::getTickFrequency() << std::endl;
    }    
    t1 = cv::getTickCount();

    // Now fill the vector of keypoints
    // Duplicate keypoints will be filtered out after 
    // the whole Feature Detection procedure is finished
    for( int i = 0; i < kpts_par.size(); i++ )
    {
        for( int j = 0; j < kpts_par[i].size(); j++ )
        {
            kpts.push_back(kpts_par[i][j]);
        }
    }
    npoints = kpts.size();

    t2 = cv::getTickCount();
    if( verbosity == true )
    {
        std::cout << "--> Fill the vector of keypoints time: "<< 1000.0*(t2-t1)/cv::getTickFrequency() << ". kpts size: " << kpts.size() << std::endl;
    }    

}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method is called by the thread which is responsible of finding extrema
 * at a given nonlinear scale level
 * @param level Index in the nonlinear scale space evolution
*/
void KAZE::Find_Extremum_Threading(int level)
{
    float value = 0.0, smax = 3.0;
    bool is_extremum = false;

    int border = fRound(smax * evolution[level].esigma) + 1;
    int ix = border, jx = border;
    while (ix < img_height-border)
    {
        jx = border;
        while (jx < img_width-border)
        {
            is_extremum = false;
            value = *(evolution[level].Ldet.ptr<float>(ix)+jx);

            // Filter the points with the detector threshold
            if( value > dthreshold && value >= DEFAULT_MIN_DETECTOR_THRESHOLD )
            {
                if( value >= *(evolution[level].Ldet.ptr<float>(ix)+jx-1) )
                {
                    // First check on the same scale
                    if( Check_Maximum_Neighbourhood(evolution[level].Ldet,1,value,ix,jx,1))
                    {
                            // Now check on the lower scale
                            if( Check_Maximum_Neighbourhood(evolution[level-1].Ldet,1,value,ix,jx,0) )
                            {
                                // Now check on the upper scale
                                if( Check_Maximum_Neighbourhood(evolution[level+1].Ldet,1,value,ix,jx,0) )
                                {
                                    is_extremum = true;
                                }
                            }
                        }                            
                    }
            }
                    
            // Add the point of interest!!
            if( is_extremum == true )
            {
                Ipoint point;
                point.xf = jx;    point.yf = ix;
                point.x = jx;    point.y = ix;
                point.dresponse = fabs(value);
                point.scale = evolution[level].esigma;
                point.sigma_size = evolution[level].sigma_size;
                point.tevolution = evolution[level].etime;
                point.octave = evolution[level].octave;
                point.sublevel = evolution[level].sublevel;
                point.level = level;
                point.descriptor_mode = descriptor_mode;
                point.angle = 0.0;
                    
                // Set the sign of the laplacian
                if( (*(evolution[level].Lxx.ptr<float>(ix)+jx) + *(evolution[level].Lyy.ptr<float>(ix)+jx)) > 0 )
                {
                    point.laplacian = 0;
                }
                else
                {
                    point.laplacian = 1;
                }
                    
                kpts_par[level-1].push_back(point);
            }
            jx++;
        }
        ix++;
    }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method performs subpixel refinement of the detected keypoints
*/
void KAZE::Do_Subpixel_Refinement(std::vector<Ipoint> &keypts)
{

    float Dx = 0.0, Dy = 0.0, Ds = 0.0, dsc = 0.0;
    float Dxx = 0.0, Dyy = 0.0, Dss = 0.0, Dxy = 0.0, Dxs = 0.0, Dys = 0.0;
    int x = 0, y = 0, step = 1;
    cv::Mat A = cv::Mat::zeros(3,3,CV_32F);
    cv::Mat b = cv::Mat::zeros(3,1,CV_32F);
    cv::Mat dst = cv::Mat::zeros(3,1,CV_32F);
    
    
    int64 t1 = cv::getTickCount();

	for( unsigned int i = 0; i < keypts.size(); i++ )
    {
         x = keypts[i].x;
         y = keypts[i].y;
             
         // Compute the gradient
         Dx = (1.0/(2.0*step))*(*(evolution[keypts[i].level].Ldet.ptr<float>(y)+x+step)
                               -*(evolution[keypts[i].level].Ldet.ptr<float>(y)+x-step));
         Dy = (1.0/(2.0*step))*(*(evolution[keypts[i].level].Ldet.ptr<float>(y+step)+x)
                               -*(evolution[keypts[i].level].Ldet.ptr<float>(y-step)+x));
         Ds = 0.5*(*(evolution[keypts[i].level+1].Ldet.ptr<float>(y)+x)
                  -*(evolution[keypts[i].level-1].Ldet.ptr<float>(y)+x));

         // Compute the Hessian
         Dxx = (1.0/(step*step))*(*(evolution[keypts[i].level].Ldet.ptr<float>(y)+x+step)
                                + *(evolution[keypts[i].level].Ldet.ptr<float>(y)+x-step)
                                -2.0*(*(evolution[keypts[i].level].Ldet.ptr<float>(y)+x)));

         Dyy = (1.0/(step*step))*(*(evolution[keypts[i].level].Ldet.ptr<float>(y+step)+x)
                                + *(evolution[keypts[i].level].Ldet.ptr<float>(y-step)+x)
                                -2.0*(*(evolution[keypts[i].level].Ldet.ptr<float>(y)+x)));

         Dss = *(evolution[keypts[i].level+1].Ldet.ptr<float>(y)+x)
               + *(evolution[keypts[i].level-1].Ldet.ptr<float>(y)+x)
              -2.0*(*(evolution[keypts[i].level].Ldet.ptr<float>(y)+x));

         Dxy = (1.0/(4.0*step))*(*(evolution[keypts[i].level].Ldet.ptr<float>(y+step)+x+step)
                               +(*(evolution[keypts[i].level].Ldet.ptr<float>(y-step)+x-step)))
               -(1.0/(4.0*step))*(*(evolution[keypts[i].level].Ldet.ptr<float>(y-step)+x+step)
                                +(*(evolution[keypts[i].level].Ldet.ptr<float>(y+step)+x-step)));

         Dxs = (1.0/(4.0*step))*(*(evolution[keypts[i].level+1].Ldet.ptr<float>(y)+x+step)
                               +(*(evolution[keypts[i].level-1].Ldet.ptr<float>(y)+x-step)))
              -(1.0/(4.0*step))*(*(evolution[keypts[i].level+1].Ldet.ptr<float>(y)+x-step)
                               +(*(evolution[keypts[i].level-1].Ldet.ptr<float>(y)+x+step)));

         Dys = (1.0/(4.0*step))*(*(evolution[keypts[i].level+1].Ldet.ptr<float>(y+step)+x)
                               +(*(evolution[keypts[i].level-1].Ldet.ptr<float>(y-step)+x)))
              -(1.0/(4.0*step))*(*(evolution[keypts[i].level+1].Ldet.ptr<float>(y-step)+x)
                               +(*(evolution[keypts[i].level-1].Ldet.ptr<float>(y+step)+x)));

         // Solve the linear system
         *(A.ptr<float>(0)) = Dxx;
         *(A.ptr<float>(1)+1) = Dyy;
         *(A.ptr<float>(2)+2) = Dss;

         *(A.ptr<float>(0)+1) = *(A.ptr<float>(1)) = Dxy;
         *(A.ptr<float>(0)+2) = *(A.ptr<float>(2)) = Dxs;
         *(A.ptr<float>(1)+2) = *(A.ptr<float>(2)+1) = Dys;

         *(b.ptr<float>(0)) = -Dx;
         *(b.ptr<float>(1)) = -Dy;
         *(b.ptr<float>(2)) = -Ds;

         cv::solve(A,b,dst,cv::DECOMP_LU);

         if( fabs(*(dst.ptr<float>(0))) <= 1.0
             && fabs(*(dst.ptr<float>(1))) <= 1.0 
             && fabs(*(dst.ptr<float>(2))) <= 1.0 )
         {             
             keypts[i].xf += *(dst.ptr<float>(0));
             keypts[i].yf += *(dst.ptr<float>(1));
             keypts[i].x = fRound(keypts[i].xf);
             keypts[i].y = fRound(keypts[i].yf);

             dsc = keypts[i].octave + (keypts[i].sublevel+*(dst.ptr<float>(2)))/((float)(DEFAULT_NSUBLEVELS));
             keypts[i].scale = soffset*pow((float)2.0,dsc);
         }
         // Delete the point since its not stable
         else
         {
             keypts[i].dresponse = 0;  // Keypoints with zero response will be filtered out
         }
    }        
    int64 t2 = cv::getTickCount();
    tsubpixel = 1000.0*(t2-t1) / cv::getTickFrequency();
    if( verbosity == true )
    {
        std::cout << "-> Subpixel refinement done. Execution time (ms): " << tsubpixel << std::endl;
    }

}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method performs feature suppression based on 2D distance
 * @param kpts Vector of keypoints
 * @param mdist Maximum distance in pixels
*/
void KAZE::Feature_Suppression_Distance(std::vector<Ipoint> &kpts, float mdist)
{
    std::vector<Ipoint> aux;
    std::vector<unsigned int> to_delete;
    float dist = 0.0, x1 = 0.0, y1 = 0.0, x2 = 0.0, y2 = 0.0;
    bool found = false;
    
    for( unsigned int i = 0; i < kpts.size(); i++ )
    {
         x1 = kpts[i].xf;
         y1 = kpts[i].yf;
         
         for( unsigned int j = i+1; j < kpts.size(); j++ )
         {
             x2 = kpts[j].xf;
             y2 = kpts[j].yf;
             
             dist = sqrt(pow(x1-x2,2)+pow(y1-y2,2));
             
             if( dist < mdist )
             {
                  if( fabs(kpts[i].dresponse) >= fabs(kpts[j].dresponse) )
                  {
                      to_delete.push_back(j);
                  }
                  else
                  {
                      to_delete.push_back(i);
                      break;
                  }
             }             
         }
    }
    
    for( unsigned int i = 0; i < kpts.size(); i++ )
    {
         found = false;
         
         for( unsigned int j = 0; j < to_delete.size(); j++ )
         {
             if( i == to_delete[j] )
             {
                 found = true;
                 break;
             }
         }
         
         if( found == false )
         {
             aux.push_back(kpts[i]);
         }
    }
    
    kpts.clear();
    kpts = aux;
    aux.clear();
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method  computes the set of descriptors through the nonlinear scale space
 * @param kpts Vector of keypoints
*/
void KAZE::Feature_Description(std::vector<Ipoint> &kpts)
{    
    if( verbosity == true )
    {
        std::cout << "\n> Computing feature descriptors. " << std::endl;
    }

    int64 t1 = cv::getTickCount();

    // It is not necessary to compute the orientation
    if( use_upright == true )
    {
        // Compute the descriptor
        if( use_extended == false )
        {
            if( descriptor_mode == 0 )
            {
                #pragma omp parallel for
                for( int i = 0; i < kpts.size(); i++ )
                {
                    kpts[i].angle = 0.0;
                    Get_SURF_Upright_Descriptor_64(kpts[i]);
                }
            }
            else if( descriptor_mode == 1 )
            {
                #pragma omp parallel for
                for( int i = 0; i < kpts.size(); i++ )
                {
                    kpts[i].angle = 0.0;
                    Get_MSURF_Upright_Descriptor_64(kpts[i]);
                }
            }
            else if( descriptor_mode == 2 )
            {
                #pragma omp parallel for
                for( int i = 0; i < kpts.size(); i++ )
                {
                    kpts[i].angle = 0.0;
                    Get_GSURF_Upright_Descriptor_64(kpts[i]);
                }
            }
        }
        else
        {
            if( descriptor_mode == 0 )
            {
                #pragma omp parallel for
                for( int i = 0; i < kpts.size(); i++ )
                {
                    kpts[i].angle = 0.0;
                    Get_SURF_Upright_Descriptor_128(kpts[i]);
                }
            }
            else if( descriptor_mode == 1 )
            {
                #pragma omp parallel for
                for( int i = 0; i < kpts.size(); i++ )
                {
                    kpts[i].angle = 0.0;
                    Get_MSURF_Upright_Descriptor_128(kpts[i]);
                }
            }
            else if( descriptor_mode == 2 )
            {
                #pragma omp parallel for
                for( int i = 0; i < kpts.size(); i++ )
                {
                    kpts[i].angle = 0.0;
                    Get_GSURF_Upright_Descriptor_128(kpts[i]);
                }
            }
        }
    }
    else
    {
        // Compute the descriptor
        if( use_extended == false )
        {
            if( descriptor_mode == 0 )
            {
                #pragma omp parallel for
                for( int i = 0; i < kpts.size(); i++ )
                {
                    Compute_Main_Orientation_SURF(kpts[i]);
                    Get_SURF_Descriptor_64(kpts[i]);
                }
            }
            else if( descriptor_mode == 1 )
            {
                #pragma omp parallel for
                for( int i = 0; i < kpts.size(); i++ )
                {
                    Compute_Main_Orientation_SURF(kpts[i]);
                    Get_MSURF_Descriptor_64(kpts[i]);
                }
            }
            else if( descriptor_mode == 2 )
            {
                #pragma omp parallel for
                for( int i = 0; i < kpts.size(); i++ )
                {
                    Compute_Main_Orientation_SURF(kpts[i]);
                    Get_GSURF_Descriptor_64(kpts[i]);
                }
            }
        }
        else
        {
            if( descriptor_mode == 0 )
            {
                #pragma omp parallel for
                for( int i = 0; i < kpts.size(); i++ )
                {
                    Compute_Main_Orientation_SURF(kpts[i]);
                    Get_SURF_Descriptor_128(kpts[i]);
                }
            }
            else if( descriptor_mode == 1 )
            {
                #pragma omp parallel for
                for( int i = 0; i < kpts.size(); i++ )
                {
                    Compute_Main_Orientation_SURF(kpts[i]);
                    Get_MSURF_Descriptor_128(kpts[i]);
                }
            }
            else if( descriptor_mode == 2 )
            {
                #pragma omp parallel for
                for( int i = 0; i < kpts.size(); i++ )
                {
                    Compute_Main_Orientation_SURF(kpts[i]);
                    Get_GSURF_Descriptor_128(kpts[i]);
                }
            }
        }
    }    
    
    int64 t2 = cv::getTickCount();
    tdescriptor = 1000.0*(t2-t1) / cv::getTickFrequency();
    if( verbosity == true )
    {
        std::cout << "> Computed feature descriptors. Execution time (ms): " << tdescriptor << std::endl;
    }

}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the main orientation for a given keypoint
 * @param kpt Input keypoint
 * @note The orientation is computed using a similar approach as described in the
 * original SURF method. See Bay et al., Speeded Up Robust Features, ECCV 2006
*/
void KAZE::Compute_Main_Orientation_SURF(Ipoint &kpt)
{
    int ix = 0, iy = 0, idx = 0, s = 0;
    unsigned int level = kpt.level;
    float xf = 0.0, yf = 0.0, gweight = 0.0;
    std::vector<float> resX(109), resY(109), Ang(109); // 109 is the maximum grids of size 1 in a circle of radius 6

    // Variables for computing the dominant direction 
    float sumX = 0.0, sumY = 0.0, bestX = 0.0, bestY = 0.0, max = 0.0, ang1 = 0.0, ang2 = 0.0;

    // Get the information from the keypoint
    xf = kpt.xf;
    yf = kpt.yf;
    s = kpt.scale;

    // Calculate derivatives responses for points within radius of 6*scale
    for(int i = -6; i <= 6; ++i) 
    {
        for(int j = -6; j <= 6; ++j) 
        {
            if(i*i + j*j < 36) // the grid is in the circle
            {
                iy = fRound(yf + j*s);
                ix = fRound(xf + i*s);
                
                if( iy >= 0 && iy < img_height && ix >= 0 && ix < img_width )
                {
                    gweight = gaussian(iy-yf,ix-xf,3.5*s);
                    resX[idx] = gweight*(*(evolution[level].Lx.ptr<float>(iy)+ix));
                    resY[idx] = gweight*(*(evolution[level].Ly.ptr<float>(iy)+ix));
                    Ang[idx] = Get_Angle(resX[idx],resY[idx]);
                }
                else
                {
                    resX[idx] = 0.0;
                    resY[idx] = 0.0;
                    Ang[idx] = 0.0;
                }
                
                ++idx;
            }
        }
    }

  // Loop slides pi/3 window around feature point
  for( ang1 = 0; ang1 < M2_PI;  ang1+=0.15f)
  {
    ang2 =(ang1+PI/3.0f > M2_PI ? ang1-5.0f*PI/3.0f : ang1+PI/3.0f);
    sumX = sumY = 0.f; 
    
    for( unsigned int k = 0; k < Ang.size(); ++k) 
    {
        // Get angle from the x-axis of the sample point
        const float & ang = Ang[k];

        // Determine whether the point is within the window
        if( ang1 < ang2 && ang1 < ang && ang < ang2) 
        {
            sumX+=resX[k];  
            sumY+=resY[k];
        } 
        else if (ang2 < ang1 && 
        ((ang > 0 && ang < ang2) || (ang > ang1 && ang < M2_PI) )) 
        {
            sumX+=resX[k];  
            sumY+=resY[k];
        }
    }

    // if the vector produced from this window is longer than all 
    // previous vectors then this forms the new dominant direction
    float sumxy = sumX*sumX + sumY*sumY;
    if( sumxy > max ) 
    {
        // store largest orientation
        max = sumxy;
        bestX = sumX, bestY = sumY;
    }
  }

  kpt.angle = Get_Angle(bestX, bestY);

}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the upright descriptor (no rotation invariant)
 * of the provided keypoint
 * @param kpt Input keypoint
 * @note Rectangular grid of 20 s x 20 s. Descriptor Length 64. No additional
 * Gaussian weighting is performed. The descriptor is inspired from Bay et al.,
 * Speeded Up Robust Features, ECCV, 2006
*/
void KAZE::Get_SURF_Upright_Descriptor_64(Ipoint &kpt)
{
  float scale = 0.0, dx = 0.0, dy = 0.0, mdx = 0.0, mdy = 0.0;
  float rx = 0.0, ry = 0.0, len = 0.0, xf = 0.0, yf = 0.0, sample_x = 0.0, sample_y = 0.0;
  float fx = 0.0, fy = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
  int x1 = 0, y1 = 0, x2 = 0, y2 = 0, sample_step = 0, pattern_size = 0, dcount = 0;
  int dsize = 0, level = 0;

  // Set the descriptor size and the sample and pattern sizes
  dsize = kpt.descriptor_size = 64;
  sample_step = 5;
  pattern_size = 10;

  // Get the information from the keypoint
  yf = kpt.yf;
  xf = kpt.xf;
  scale = kpt.scale;
  level = kpt.level;

  // Allocate the memory for the vector
  kpt.descriptor = vector<float>(kpt.descriptor_size);
  
  // Calculate descriptor for this interest point
  for(int i = -pattern_size; i < pattern_size; i+=sample_step)
  {
    for(int j = -pattern_size; j < pattern_size; j+=sample_step)
    {
         dx=dy=mdx=mdy=0.0;
      
          for(float k = i; k < i + sample_step; k+=0.5)
          {
            for(float l = j; l < j + sample_step; l+=0.5)
            {
                sample_y = k*scale + yf;
                sample_x = l*scale + xf;

                y1 = (int)(sample_y-.5);
                x1 = (int)(sample_x-.5);

                Check_Descriptor_Limits(x1,y1,img_width,img_height);
                
                y2 = (int)(sample_y+.5);
                x2 = (int)(sample_x+.5);
    
                Check_Descriptor_Limits(x2,y2,img_width,img_height);
                
                fx = sample_x-x1;
                fy = sample_y-y1;

                res1 = *(evolution[level].Lx.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Lx.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Lx.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Lx.ptr<float>(y2)+x2);
                rx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;
                
                res1 = *(evolution[level].Ly.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Ly.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Ly.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Ly.ptr<float>(y2)+x2);
                ry = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;
    
                // Sum the derivatives to the cumulative descriptor
                dx += rx;
                dy += ry;
                mdx += fabs(rx);
                mdy += fabs(ry);
            }
          }
      
      // Add the values to the descriptor vector
      kpt.descriptor[dcount++] = dx;
      kpt.descriptor[dcount++] = dy;
      kpt.descriptor[dcount++] = mdx;
      kpt.descriptor[dcount++] = mdy;

      // Store the current length^2 of the vector
      len += dx*dx + dy*dy + mdx*mdx + mdy*mdy;
    }
  }

  // convert to unit vector
  len = sqrt(len);

  for(int i = 0; i < dsize; i++)
  {
      kpt.descriptor[i] /= len;
  }

  if( USE_CLIPPING_NORMALIZATION == true )
  {
      Clipping_Descriptor(kpt,CLIPPING_NORMALIZATION_NITER,CLIPPING_NORMALIZATION_RATIO);
  }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the descriptor of the provided keypoint given the
 * main orientation
 * @param kpt Input keypoint
 * @note Rectangular grid of 20 s x 20 s. Descriptor Length 64. No additional
 * Gaussian weighting is performed. The descriptor is inspired from Bay et al.,
 * Speeded Up Robust Features, ECCV, 2006
*/
void KAZE::Get_SURF_Descriptor_64(Ipoint &kpt)
{
  float scale = 0.0, dx = 0.0, dy = 0.0, mdx = 0.0, mdy = 0.0;
  float rx = 0.0, ry = 0.0, rrx = 0.0, rry = 0.0, len = 0.0, xf = 0.0, yf = 0.0;
  float sample_x = 0.0, sample_y = 0.0, co = 0.0, si = 0.0, angle = 0.0;
  float fx = 0.0, fy = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
  int x1 = 0, y1 = 0, x2 = 0, y2 = 0, sample_step = 0, pattern_size = 0, dcount = 0;
  int dsize = 0, level = 0;

  // Set the descriptor size and the sample and pattern sizes
  dsize = kpt.descriptor_size = 64;
  sample_step = 5;
  pattern_size = 10;

  // Get the information from the keypoint
  yf = kpt.yf;
  xf = kpt.xf;
  scale = kpt.scale;
  angle = kpt.angle;
  level = kpt.level;
  co = cos(angle);
  si = sin(angle);

  // Allocate the memory for the vector
  kpt.descriptor = vector<float>(kpt.descriptor_size);
  
  // Calculate descriptor for this interest point
  for(int i = -pattern_size; i < pattern_size; i+=sample_step)
  {
    for(int j = -pattern_size; j < pattern_size; j+=sample_step)
    {
         dx=dy=mdx=mdy=0.0;
      
          for(float k = i; k < i + sample_step; k+=0.5)
          {
            for(float l = j; l < j + sample_step; l+=0.5)
            {
                // Get the coordinates of the sample point on the rotated axis
                sample_y = yf + (l*scale*co + k*scale*si);
                sample_x = xf + (-l*scale*si + k*scale*co);
                
                y1 = (int)(sample_y-.5);
                x1 = (int)(sample_x-.5);

                Check_Descriptor_Limits(x1,y1,img_width,img_height);
                
                y2 = (int)(sample_y+.5);
                x2 = (int)(sample_x+.5);
    
                Check_Descriptor_Limits(x2,y2,img_width,img_height);
                
                fx = sample_x-x1;
                fy = sample_y-y1;


                res1 = *(evolution[level].Lx.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Lx.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Lx.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Lx.ptr<float>(y2)+x2);
                rx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;
                
                res1 = *(evolution[level].Ly.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Ly.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Ly.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Ly.ptr<float>(y2)+x2);
                ry = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                // Get the x and y derivatives on the rotated axis
                rry = rx*co + ry*si;
                rrx = -rx*si + ry*co;

                // Sum the derivatives to the cumulative descriptor
                dx += rrx;
                dy += rry;
                mdx += fabs(rrx);
                mdy += fabs(rry);
            }
        }

      // Add the values to the descriptor vector
      kpt.descriptor[dcount++] = dx;
      kpt.descriptor[dcount++] = dy;
      kpt.descriptor[dcount++] = mdx;
      kpt.descriptor[dcount++] = mdy;

      // Store the current length^2 of the vector
      len += dx*dx + dy*dy + mdx*mdx + mdy*mdy;
    }
  }

  // convert to unit vector
  len = sqrt(len);

  for(int i = 0; i < dsize; i++)
  {
      kpt.descriptor[i] /= len;
  }

  if( USE_CLIPPING_NORMALIZATION == true )
  {
      Clipping_Descriptor(kpt,CLIPPING_NORMALIZATION_NITER,CLIPPING_NORMALIZATION_RATIO);
  }
  
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the upright descriptor (not rotation invariant) of
 * the provided keypoint
 * @param kpt Input keypoint
 * @note Rectangular grid of 24 s x 24 s. Descriptor Length 64. The descriptor is inspired
 * from Agrawal et al., CenSurE: Center Surround Extremas for Realtime Feature Detection and Matching,
 * ECCV 2008
*/
void KAZE::Get_MSURF_Upright_Descriptor_64(Ipoint &kpt)
{
  float scale = 0.0, dx = 0.0, dy = 0.0, mdx = 0.0, mdy = 0.0, gauss_s1 = 0.0, gauss_s2 = 0.0;
  float rx = 0.0, ry = 0.0, len = 0.0, xf = 0.0, yf = 0.0, ys = 0.0, xs = 0.0;
  float sample_x = 0.0, sample_y = 0.0;
  int x1 = 0, y1 = 0, sample_step = 0, pattern_size = 0;
  int x2 = 0, y2 = 0, kx = 0, ky = 0, i = 0, j = 0, dcount = 0;
  float fx = 0.0, fy = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
  int dsize = 0, level = 0;
   
  // Subregion centers for the 4x4 gaussian weighting
  float cx = -0.5, cy = 0.5; 
  
  // Set the descriptor size and the sample and pattern sizes
  dsize = kpt.descriptor_size = 64;
  sample_step = 5;
  pattern_size = 12;

  // Get the information from the keypoint
  yf = kpt.yf;
  xf = kpt.xf;
  level = kpt.level;
  scale = kpt.scale;
  
  // Allocate the memory for the vector
  kpt.descriptor = vector<float>(kpt.descriptor_size);
  
  i = -8;
  
  // Calculate descriptor for this interest point
  // Area of size 24 s x 24 s
  while(i < pattern_size)
  {
     j = -8;
     i = i-4;

     cx += 1.0;
     cy = -0.5;
    
     while(j < pattern_size)
     {
        dx=dy=mdx=mdy=0.0;
        cy += 1.0;
        j = j-4;
        
        ky = i + sample_step;
        kx = j + sample_step;

        ys = yf + (ky*scale);    
        xs = xf + (kx*scale);
        
        for(int k = i; k < i+9; k++)
        {
            for (int l = j; l < j+9; l++)
            {
                sample_y = k*scale + yf;
                sample_x = l*scale + xf;

                //Get the gaussian weighted x and y responses
                gauss_s1 = gaussian(xs-sample_x,ys-sample_y,2.50*scale);
                
                y1 = (int)(sample_y-.5);
                x1 = (int)(sample_x-.5);

                Check_Descriptor_Limits(x1,y1,img_width,img_height);
                
                y2 = (int)(sample_y+.5);
                x2 = (int)(sample_x+.5);
    
                Check_Descriptor_Limits(x2,y2,img_width,img_height);
                
                fx = sample_x-x1;
                fy = sample_y-y1;

                res1 = *(evolution[level].Lx.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Lx.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Lx.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Lx.ptr<float>(y2)+x2);
                rx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;
                                    
                res1 = *(evolution[level].Ly.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Ly.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Ly.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Ly.ptr<float>(y2)+x2);
                ry = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                rx = gauss_s1*rx;
                ry = gauss_s1*ry;
                
                // Sum the derivatives to the cumulative descriptor
                dx += rx;
                dy += ry;
                mdx += fabs(rx);
                mdy += fabs(ry);
            }
        }
        
        // Add the values to the descriptor vector
        gauss_s2 = gaussian(cx-2.0f,cy-2.0f,1.5f);

        kpt.descriptor[dcount++] = dx*gauss_s2;
        kpt.descriptor[dcount++] = dy*gauss_s2;
        kpt.descriptor[dcount++] = mdx*gauss_s2;
        kpt.descriptor[dcount++] = mdy*gauss_s2;
                
        len += (dx*dx + dy*dy + mdx*mdx + mdy*mdy)*gauss_s2*gauss_s2;

        j += 9;
    }

        i += 9;
  }

  // convert to unit vector
  len = sqrt(len);

  for(int i = 0; i < dsize; i++)
  {
      kpt.descriptor[i] /= len;
  }

  if( USE_CLIPPING_NORMALIZATION == true )
  {
      Clipping_Descriptor(kpt,CLIPPING_NORMALIZATION_NITER,CLIPPING_NORMALIZATION_RATIO);
  }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the descriptor of the provided keypoint given the
 * main orientation of the keypoint
 * @param kpt Input keypoint
 * @note Rectangular grid of 24 s x 24 s. Descriptor Length 64. The descriptor is inspired
 * from Agrawal et al., CenSurE: Center Surround Extremas for Realtime Feature Detection and Matching,
 * ECCV 2008
*/
void KAZE::Get_MSURF_Descriptor_64(Ipoint &kpt)
{
  float scale = 0.0, dx = 0.0, dy = 0.0, mdx = 0.0, mdy = 0.0, gauss_s1 = 0.0, gauss_s2 = 0.0;
  float rx = 0.0, ry = 0.0, rrx = 0.0, rry = 0.0, len = 0.0, xf = 0.0, yf = 0.0, ys = 0.0, xs = 0.0;
  float sample_x = 0.0, sample_y = 0.0, co = 0.0, si = 0.0, angle = 0.0;
  float fx = 0.0, fy = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
  int x1 = 0, y1 = 0, x2 = 0, y2 = 0, sample_step = 0, pattern_size = 0;
  int kx = 0, ky = 0, i = 0, j = 0, dcount = 0;
  int dsize = 0, level = 0;
  
  // Subregion centers for the 4x4 gaussian weighting
  float cx = -0.5, cy = 0.5; 
  
  // Set the descriptor size and the sample and pattern sizes
  dsize = kpt.descriptor_size = 64;
  sample_step = 5;
  pattern_size = 12;

  // Get the information from the keypoint
  yf = kpt.yf;
  xf = kpt.xf;
  scale = kpt.scale;
  angle = kpt.angle;
  level = kpt.level;
  co = cos(angle);
  si = sin(angle);
  
  // Allocate the memory for the vector
  kpt.descriptor = vector<float>(kpt.descriptor_size);
  
  i = -8;
  
  // Calculate descriptor for this interest point
  // Area of size 24 s x 24 s
  while(i < pattern_size)
  {
     j = -8;
     i = i-4;

     cx += 1.0;
     cy = -0.5;
    
     while(j < pattern_size)
     {
        dx=dy=mdx=mdy=0.0;
        cy += 1.0;
        j = j - 4;
        
        ky = i + sample_step;
        kx = j + sample_step;

        xs = xf + (-kx*scale*si + ky*scale*co);
        ys = yf + (kx*scale*co + ky*scale*si);
        
        for (int k = i; k < i + 9; ++k)
        {
            for (int l = j; l < j + 9; ++l)
            {
                // Get coords of sample point on the rotated axis
                sample_y = yf + (l*scale*co + k*scale*si);
                sample_x = xf + (-l*scale*si + k*scale*co);
          
                // Get the gaussian weighted x and y responses
                gauss_s1 = gaussian(xs-sample_x,ys-sample_y,2.5*scale);

                y1 = fRound(sample_y-.5);
                x1 = fRound(sample_x-.5);

                Check_Descriptor_Limits(x1,y1,img_width,img_height);

                y2 = fRound(sample_y+.5);
                x2 = fRound(sample_x+.5);
    
                Check_Descriptor_Limits(x2,y2,img_width,img_height);
    
                fx = sample_x-x1;
                fy = sample_y-y1;

                res1 = *(evolution[level].Lx.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Lx.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Lx.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Lx.ptr<float>(y2)+x2);
                rx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;
                                    
                res1 = *(evolution[level].Ly.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Ly.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Ly.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Ly.ptr<float>(y2)+x2);
                ry = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                // Get the x and y derivatives on the rotated axis
                rry = gauss_s1*(rx*co + ry*si);
                rrx = gauss_s1*(-rx*si + ry*co);

                // Sum the derivatives to the cumulative descriptor
                dx += rrx;
                dy += rry;
                mdx += fabs(rrx);
                mdy += fabs(rry);
            }
        }
        
        // Add the values to the descriptor vector
        gauss_s2 = gaussian(cx-2.0f,cy-2.0f,1.5f);
        kpt.descriptor[dcount++] = dx*gauss_s2;
        kpt.descriptor[dcount++] = dy*gauss_s2;
        kpt.descriptor[dcount++] = mdx*gauss_s2;
        kpt.descriptor[dcount++] = mdy*gauss_s2;
                
        len += (dx*dx + dy*dy + mdx*mdx + mdy*mdy)*gauss_s2*gauss_s2;

        j += 9;
    }

        i += 9;
  }

  // convert to unit vector
  len = sqrt(len);

  for(int i = 0; i < dsize; i++)
  {
      kpt.descriptor[i] /= len;
  }

  if( USE_CLIPPING_NORMALIZATION == true )
  {
      Clipping_Descriptor(kpt,CLIPPING_NORMALIZATION_NITER,CLIPPING_NORMALIZATION_RATIO);
  }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the upright G-SURF descriptor of the provided keypoint
 * given the main orientation
 * @param kpt Input keypoint
 * @note Rectangular grid of 20 s x 20 s. Descriptor Length 64. No additional
 * G-SURF descriptor as described in Pablo F. Alcantarilla, Luis M. Bergasa and
 * Andrew J. Davison, Gauge-SURF Descriptors, Image and Vision Computing 31(1), 2013
*/
void KAZE::Get_GSURF_Upright_Descriptor_64(Ipoint &kpt)
{
  float scale = 0.0, dx = 0.0, dy = 0.0, mdx = 0.0, mdy = 0.0;
  float rx = 0.0, ry = 0.0, rxx = 0.0, rxy = 0.0, ryy = 0.0, len = 0.0, xf = 0.0, yf = 0.0;
  float sample_x = 0.0, sample_y = 0.0;
  float fx = 0.0, fy = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
  float lvv = 0.0, lww = 0.0, modg = 0.0;
  int x1 = 0, y1 = 0, x2 = 0, y2 = 0, sample_step = 0, pattern_size = 0, dcount = 0;
  int dsize = 0, level = 0;

  // Set the descriptor size and the sample and pattern sizes
  dsize = kpt.descriptor_size = 64;
  sample_step = 5;
  pattern_size = 10;

  // Get the information from the keypoint
  yf = kpt.yf;
  xf = kpt.xf;
  scale = kpt.scale;
  level = kpt.level;

  // Allocate the memory for the vector
  kpt.descriptor = vector<float>(kpt.descriptor_size);

  // Calculate descriptor for this interest point
  for(int i = -pattern_size; i < pattern_size; i+=sample_step)
  {
    for(int j = -pattern_size; j < pattern_size; j+=sample_step)
    {
         dx=dy=mdx=mdy=0.0;

          for(float k = i; k < i + sample_step; k+=0.5)
          {
            for(float l = j; l < j + sample_step; l+=0.5)
            {
                // Get the coordinates of the sample point on the rotated axis
                sample_y = yf + l*scale;
                sample_x = xf + k*scale;

                y1 = (int)(sample_y-.5);
                x1 = (int)(sample_x-.5);

                Check_Descriptor_Limits(x1,y1,img_width,img_height);

                y2 = (int)(sample_y+.5);
                x2 = (int)(sample_x+.5);

                Check_Descriptor_Limits(x2,y2,img_width,img_height);

                fx = sample_x-x1;
                fy = sample_y-y1;

                res1 = *(evolution[level].Lx.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Lx.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Lx.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Lx.ptr<float>(y2)+x2);
                rx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                res1 = *(evolution[level].Ly.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Ly.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Ly.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Ly.ptr<float>(y2)+x2);
                ry = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                modg = pow(rx,2) + pow(ry,2);

                if( modg != 0.0 )
                {
                    res1 = *(evolution[level].Lxx.ptr<float>(y1)+x1);
                    res2 = *(evolution[level].Lxx.ptr<float>(y1)+x2);
                    res3 = *(evolution[level].Lxx.ptr<float>(y2)+x1);
                    res4 = *(evolution[level].Lxx.ptr<float>(y2)+x2);
                    rxx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                    res1 = *(evolution[level].Lxy.ptr<float>(y1)+x1);
                    res2 = *(evolution[level].Lxy.ptr<float>(y1)+x2);
                    res3 = *(evolution[level].Lxy.ptr<float>(y2)+x1);
                    res4 = *(evolution[level].Lxy.ptr<float>(y2)+x2);
                    rxy = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                    res1 = *(evolution[level].Lyy.ptr<float>(y1)+x1);
                    res2 = *(evolution[level].Lyy.ptr<float>(y1)+x2);
                    res3 = *(evolution[level].Lyy.ptr<float>(y2)+x1);
                    res4 = *(evolution[level].Lyy.ptr<float>(y2)+x2);
                    ryy = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                    // Lww = (Lx^2 * Lxx + 2*Lx*Lxy*Ly + Ly^2*Lyy) / (Lx^2 + Ly^2)
                    lww = (pow(rx,2)*rxx + 2.0*rx*rxy*ry + pow(ry,2)*ryy) / (modg);

                    // Lvv = (-2*Lx*Lxy*Ly + Lxx*Ly^2 + Lx^2*Lyy) / (Lx^2 + Ly^2)
                    lvv = (-2.0*rx*rxy*ry + rxx*pow(ry,2) + pow(rx,2)*ryy) /(modg);
                }
                else
                {
                      lww = 0.0;
                      lvv = 0.0;
                }

                // Sum the derivatives to the cumulative descriptor
                dx += lww;
                dy += lvv;
                mdx += fabs(lww);
                mdy += fabs(lvv);
            }
        }

      // Add the values to the descriptor vector
      kpt.descriptor[dcount++] = dx;
      kpt.descriptor[dcount++] = dy;
      kpt.descriptor[dcount++] = mdx;
      kpt.descriptor[dcount++] = mdy;

      // Store the current length^2 of the vector
      len += dx*dx + dy*dy + mdx*mdx + mdy*mdy;
    }
  }

  // convert to unit vector
  len = sqrt(len);

  for(int i = 0; i < dsize; i++)
  {
      kpt.descriptor[i] /= len;
  }

  if( USE_CLIPPING_NORMALIZATION == true )
  {
      Clipping_Descriptor(kpt,CLIPPING_NORMALIZATION_NITER,CLIPPING_NORMALIZATION_RATIO);
  }

}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the G-SURF descriptor of the provided keypoint given the
 * main orientation
 * @param kpt Input keypoint
 * @note Rectangular grid of 20 s x 20 s. Descriptor Length 64. No additional
 * G-SURF descriptor as described in Pablo F. Alcantarilla, Luis M. Bergasa and
 * Andrew J. Davison, Gauge-SURF Descriptors, Image and Vision Computing 31(1), 2013
*/
void KAZE::Get_GSURF_Descriptor_64(Ipoint &kpt)
{
  float scale = 0.0, dx = 0.0, dy = 0.0, mdx = 0.0, mdy = 0.0;
  float rx = 0.0, ry = 0.0, rxx = 0.0, rxy = 0.0, ryy = 0.0, len = 0.0, xf = 0.0, yf = 0.0;
  float sample_x = 0.0, sample_y = 0.0, co = 0.0, si = 0.0, angle = 0.0;
  float fx = 0.0, fy = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
  float lvv = 0.0, lww = 0.0, modg = 0.0;
  int x1 = 0, y1 = 0, x2 = 0, y2 = 0, sample_step = 0, pattern_size = 0, dcount = 0;
  int dsize = 0, level = 0;

  // Set the descriptor size and the sample and pattern sizes
  dsize = kpt.descriptor_size = 64;
  sample_step = 5;
  pattern_size = 10;

  // Get the information from the keypoint
  yf = kpt.yf;
  xf = kpt.xf;
  scale = kpt.scale;
  angle = kpt.angle;
  level = kpt.level;
  co = cos(angle);
  si = sin(angle);

  // Allocate the memory for the vector
  kpt.descriptor = vector<float>(kpt.descriptor_size);

  // Calculate descriptor for this interest point
  for(int i = -pattern_size; i < pattern_size; i+=sample_step)
  {
    for(int j = -pattern_size; j < pattern_size; j+=sample_step)
    {
         dx=dy=mdx=mdy=0.0;

          for(float k = i; k < i + sample_step; k+=0.5)
          {
            for(float l = j; l < j + sample_step; l+=0.5)
            {
                // Get the coordinates of the sample point on the rotated axis
                sample_y = yf + (l*scale*co + k*scale*si);
                sample_x = xf + (-l*scale*si + k*scale*co);

                y1 = (int)(sample_y-.5);
                x1 = (int)(sample_x-.5);

                Check_Descriptor_Limits(x1,y1,img_width,img_height);

                y2 = (int)(sample_y+.5);
                x2 = (int)(sample_x+.5);

                Check_Descriptor_Limits(x2,y2,img_width,img_height);

                fx = sample_x-x1;
                fy = sample_y-y1;

                res1 = *(evolution[level].Lx.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Lx.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Lx.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Lx.ptr<float>(y2)+x2);
                rx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                res1 = *(evolution[level].Ly.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Ly.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Ly.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Ly.ptr<float>(y2)+x2);
                ry = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                modg = pow(rx,2) + pow(ry,2);

                if( modg != 0.0 )
                {
                    res1 = *(evolution[level].Lxx.ptr<float>(y1)+x1);
                    res2 = *(evolution[level].Lxx.ptr<float>(y1)+x2);
                    res3 = *(evolution[level].Lxx.ptr<float>(y2)+x1);
                    res4 = *(evolution[level].Lxx.ptr<float>(y2)+x2);
                    rxx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                    res1 = *(evolution[level].Lxy.ptr<float>(y1)+x1);
                    res2 = *(evolution[level].Lxy.ptr<float>(y1)+x2);
                    res3 = *(evolution[level].Lxy.ptr<float>(y2)+x1);
                    res4 = *(evolution[level].Lxy.ptr<float>(y2)+x2);
                    rxy = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                    res1 = *(evolution[level].Lyy.ptr<float>(y1)+x1);
                    res2 = *(evolution[level].Lyy.ptr<float>(y1)+x2);
                    res3 = *(evolution[level].Lyy.ptr<float>(y2)+x1);
                    res4 = *(evolution[level].Lyy.ptr<float>(y2)+x2);
                    ryy = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                    // Lww = (Lx^2 * Lxx + 2*Lx*Lxy*Ly + Ly^2*Lyy) / (Lx^2 + Ly^2)
                    lww = (pow(rx,2)*rxx + 2.0*rx*rxy*ry + pow(ry,2)*ryy) / (modg);

                    // Lvv = (-2*Lx*Lxy*Ly + Lxx*Ly^2 + Lx^2*Lyy) / (Lx^2 + Ly^2)
                    lvv = (-2.0*rx*rxy*ry + rxx*pow(ry,2) + pow(rx,2)*ryy) /(modg);
                }
                else
                {
                      lww = 0.0;
                      lvv = 0.0;
                }

                // Sum the derivatives to the cumulative descriptor
                dx += lww;
                dy += lvv;
                mdx += fabs(lww);
                mdy += fabs(lvv);
            }
        }

      // Add the values to the descriptor vector
      kpt.descriptor[dcount++] = dx;
      kpt.descriptor[dcount++] = dy;
      kpt.descriptor[dcount++] = mdx;
      kpt.descriptor[dcount++] = mdy;

      // Store the current length^2 of the vector
      len += dx*dx + dy*dy + mdx*mdx + mdy*mdy;
    }
  }

  // convert to unit vector
  len = sqrt(len);

  for(int i = 0; i < dsize; i++)
  {
      kpt.descriptor[i] /= len;
  }

  if( USE_CLIPPING_NORMALIZATION == true )
  {
      Clipping_Descriptor(kpt,CLIPPING_NORMALIZATION_NITER,CLIPPING_NORMALIZATION_RATIO);
  }

}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the upright extended descriptor (no rotation invariant)
 * of the provided keypoint
 * @param kpt Input keypoint
 * @note Rectangular grid of 20 s x 20 s. Descriptor Length 128. No additional
 * Gaussian weighting is performed. The descriptor is inspired from Bay et al.,
 * Speeded Up Robust Features, ECCV, 2006
*/
void KAZE::Get_SURF_Upright_Descriptor_128(Ipoint &kpt)
{
  float scale = 0.0;
  float rx = 0.0, ry = 0.0, len = 0.0, xf = 0.0, yf = 0.0, sample_x = 0.0, sample_y = 0.0;
  float fx = 0.0, fy = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
  float dxp = 0.0, dyp = 0.0, mdxp = 0.0, mdyp = 0.0;
  float dxn = 0.0, dyn = 0.0, mdxn = 0.0, mdyn = 0.0;
  int x1 = 0, y1 = 0, x2 = 0, y2 = 0, sample_step = 0, pattern_size = 0, dcount = 0;
  int dsize = 0, level = 0;

  // Set the descriptor size and the sample and pattern sizes
  dsize = kpt.descriptor_size = 128;
  sample_step = 5;
  pattern_size = 10;

  // Get the information from the keypoint
  yf = kpt.yf;
  xf = kpt.xf;
  scale = kpt.scale;
  level = kpt.level;

  // Allocate the memory for the vector
  kpt.descriptor = vector<float>(kpt.descriptor_size);

  // Calculate descriptor for this interest point
  for(int i = -pattern_size; i < pattern_size; i+=sample_step)
  {
    for(int j = -pattern_size; j < pattern_size; j+=sample_step)
    {
         dxp=dxn=mdxp=mdxn=0.0;
         dyp=dyn=mdyp=mdyn=0.0;

          for(float k = i; k < i + sample_step; k+=0.5)
          {
            for(float l = j; l < j + sample_step; l+=0.5)
            {
                sample_y = k*scale + yf;
                sample_x = l*scale + xf;

                y1 = (int)(sample_y-.5);
                x1 = (int)(sample_x-.5);

                Check_Descriptor_Limits(x1,y1,img_width,img_height);

                y2 = (int)(sample_y+.5);
                x2 = (int)(sample_x+.5);

                Check_Descriptor_Limits(x2,y2,img_width,img_height);

                fx = sample_x-x1;
                fy = sample_y-y1;

                res1 = *(evolution[level].Lx.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Lx.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Lx.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Lx.ptr<float>(y2)+x2);
                rx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                res1 = *(evolution[level].Ly.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Ly.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Ly.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Ly.ptr<float>(y2)+x2);
                ry = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                // Sum the derivatives to the cumulative descriptor
                if( ry >= 0.0 )
                {
                    dxp += rx;
                    mdxp += fabs(rx);
                }
                else
                {
                    dxn += rx;
                    mdxn += fabs(rx);
                }

                if( rx >= 0.0 )
                {
                    dyp += ry;
                    mdyp += fabs(ry);
                }
                else
                {
                    dyn += ry;
                    mdyn += fabs(ry);
                }
            }
          }

      // Add the values to the descriptor vector
      kpt.descriptor[dcount++] = dxp;
      kpt.descriptor[dcount++] = dxn;
      kpt.descriptor[dcount++] = mdxp;
      kpt.descriptor[dcount++] = mdxn;
      kpt.descriptor[dcount++] = dyp;
      kpt.descriptor[dcount++] = dyn;
      kpt.descriptor[dcount++] = mdyp;
      kpt.descriptor[dcount++] = mdyn;

      // Store the current length^2 of the vector
      len += dxp*dxp + dxn*dxn + mdxp*mdxp + mdxn*mdxn +
             dyp*dyp + dyn*dyn + mdyp*mdyp + mdyn*mdyn;
    }
  }

  // convert to unit vector
  len = sqrt(len);

  for(int i = 0; i < dsize; i++)
  {
      kpt.descriptor[i] /= len;
  }

  if( USE_CLIPPING_NORMALIZATION == true )
  {
      Clipping_Descriptor(kpt,CLIPPING_NORMALIZATION_NITER,CLIPPING_NORMALIZATION_RATIO);
  }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the extended descriptor of the provided keypoint given the
 * main orientation
 * @param kpt Input keypoint
 * @note Rectangular grid of 20 s x 20 s. Descriptor Length 128. No additional
 * Gaussian weighting is performed. The descriptor is inspired from Bay et al.,
 * Speeded Up Robust Features, ECCV, 2006
*/
void KAZE::Get_SURF_Descriptor_128(Ipoint &kpt)
{
  float scale = 0.0;
  float rx = 0.0, ry = 0.0, rrx = 0.0, rry = 0.0, len = 0.0, xf = 0.0, yf = 0.0;
  float sample_x = 0.0, sample_y = 0.0, co = 0.0, si = 0.0, angle = 0.0;
  float fx = 0.0, fy = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
  float dxp = 0.0, dyp = 0.0, mdxp = 0.0, mdyp = 0.0;
  float dxn = 0.0, dyn = 0.0, mdxn = 0.0, mdyn = 0.0;
  int x1 = 0, y1 = 0, x2 = 0, y2 = 0, sample_step = 0, pattern_size = 0, dcount = 0;
  int dsize = 0, level = 0;

  // Set the descriptor size and the sample and pattern sizes
  dsize = kpt.descriptor_size = 128;
  sample_step = 5;
  pattern_size = 10;

  // Get the information from the keypoint
  yf = kpt.yf;
  xf = kpt.xf;
  scale = kpt.scale;
  angle = kpt.angle;
  level = kpt.level;
  co = cos(angle);
  si = sin(angle);

  // Allocate the memory for the vector
  kpt.descriptor = vector<float>(kpt.descriptor_size);

  // Calculate descriptor for this interest point
  for(int i = -pattern_size; i < pattern_size; i+=sample_step)
  {
    for(int j = -pattern_size; j < pattern_size; j+=sample_step)
    {
         dxp=dxn=mdxp=mdxn=0.0;
         dyp=dyn=mdyp=mdyn=0.0;

          for(float k = i; k < i + sample_step; k+=0.5)
          {
            for(float l = j; l < j + sample_step; l+=0.5)
            {
                // Get the coordinates of the sample point on the rotated axis
                sample_y = yf + (l*scale*co + k*scale*si);
                sample_x = xf + (-l*scale*si + k*scale*co);

                y1 = (int)(sample_y-.5);
                x1 = (int)(sample_x-.5);

                Check_Descriptor_Limits(x1,y1,img_width,img_height);

                y2 = (int)(sample_y+.5);
                x2 = (int)(sample_x+.5);

                Check_Descriptor_Limits(x2,y2,img_width,img_height);

                fx = sample_x-x1;
                fy = sample_y-y1;

                res1 = *(evolution[level].Lx.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Lx.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Lx.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Lx.ptr<float>(y2)+x2);
                rx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                res1 = *(evolution[level].Ly.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Ly.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Ly.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Ly.ptr<float>(y2)+x2);
                ry = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                // Get the x and y derivatives on the rotated axis
                rry = rx*co + ry*si;
                rrx = -rx*si + ry*co;

                // Sum the derivatives to the cumulative descriptor
                if( rry >= 0.0 )
                {
                    dxp += rrx;
                    mdxp += fabs(rrx);
                }
                else
                {
                    dxn += rrx;
                    mdxn += fabs(rrx);
                }

                if( rrx >= 0.0 )
                {
                    dyp += rry;
                    mdyp += fabs(rry);
                }
                else
                {
                    dyn += rry;
                    mdyn += fabs(rry);
                }
            }
        }

        // Add the values to the descriptor vector
        kpt.descriptor[dcount++] = dxp;
        kpt.descriptor[dcount++] = dxn;
        kpt.descriptor[dcount++] = mdxp;
        kpt.descriptor[dcount++] = mdxn;
        kpt.descriptor[dcount++] = dyp;
        kpt.descriptor[dcount++] = dyn;
        kpt.descriptor[dcount++] = mdyp;
        kpt.descriptor[dcount++] = mdyn;

        // Store the current length^2 of the vector
        len += dxp*dxp + dxn*dxn + mdxp*mdxp + mdxn*mdxn +
               dyp*dyp + dyn*dyn + mdyp*mdyp + mdyn*mdyn;
    }
  }

  // convert to unit vector
  len = sqrt(len);

  for(int i = 0; i < dsize; i++)
  {
      kpt.descriptor[i] /= len;
  }

  if( USE_CLIPPING_NORMALIZATION == true )
  {
      Clipping_Descriptor(kpt,CLIPPING_NORMALIZATION_NITER,CLIPPING_NORMALIZATION_RATIO);
  }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the extended upright descriptor (not rotation invariant) of
 * the provided keypoint
 * @param kpt Input keypoint
 * @note Rectangular grid of 24 s x 24 s. Descriptor Length 128. The descriptor is inspired
 * from Agrawal et al., CenSurE: Center Surround Extremas for Realtime Feature Detection and Matching,
 * ECCV 2008
*/
void KAZE::Get_MSURF_Upright_Descriptor_128(Ipoint &kpt)
{
  float scale = 0.0, gauss_s1 = 0.0, gauss_s2 = 0.0;
  float rx = 0.0, ry = 0.0, len = 0.0, xf = 0.0, yf = 0.0, ys = 0.0, xs = 0.0;
  float sample_x = 0.0, sample_y = 0.0;
  int x1 = 0, y1 = 0, sample_step = 0, pattern_size = 0;
  int x2 = 0, y2 = 0, kx = 0, ky = 0, i = 0, j = 0, dcount = 0;
  float fx = 0.0, fy = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
  float dxp = 0.0, dyp = 0.0, mdxp = 0.0, mdyp = 0.0;
  float dxn = 0.0, dyn = 0.0, mdxn = 0.0, mdyn = 0.0;
  int dsize = 0, level = 0;

  // Subregion centers for the 4x4 gaussian weighting
  float cx = -0.5, cy = 0.5;

  // Set the descriptor size and the sample and pattern sizes
  dsize = kpt.descriptor_size = 128;
  sample_step = 5;
  pattern_size = 12;

  // Get the information from the keypoint
  yf = kpt.yf;
  xf = kpt.xf;
  level = kpt.level;
  scale = kpt.scale;

  // Allocate the memory for the vector
  kpt.descriptor = vector<float>(kpt.descriptor_size);

  i = -8;

  // Calculate descriptor for this interest point
  // Area of size 24 s x 24 s
  while(i < pattern_size)
  {
     j = -8;
     i = i-4;

     cx += 1.0;
     cy = -0.5;

     while(j < pattern_size)
     {
        dxp=dxn=mdxp=mdxn=0.0;
        dyp=dyn=mdyp=mdyn=0.0;

        cy += 1.0;
        j = j-4;

        ky = i + sample_step;
        kx = j + sample_step;

        ys = yf + (ky*scale);
        xs = xf + (kx*scale);

        for(int k = i; k < i+9; k++)
        {
            for (int l = j; l < j+9; l++)
            {
                sample_y = k*scale + yf;
                sample_x = l*scale + xf;

                //Get the gaussian weighted x and y responses
                gauss_s1 = gaussian(xs-sample_x,ys-sample_y,2.50*scale);

                y1 = (int)(sample_y-.5);
                x1 = (int)(sample_x-.5);

                Check_Descriptor_Limits(x1,y1,img_width,img_height);

                y2 = (int)(sample_y+.5);
                x2 = (int)(sample_x+.5);

                Check_Descriptor_Limits(x2,y2,img_width,img_height);

                fx = sample_x-x1;
                fy = sample_y-y1;

                res1 = *(evolution[level].Lx.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Lx.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Lx.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Lx.ptr<float>(y2)+x2);
                rx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                res1 = *(evolution[level].Ly.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Ly.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Ly.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Ly.ptr<float>(y2)+x2);
                ry = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                rx = gauss_s1*rx;
                ry = gauss_s1*ry;

                // Sum the derivatives to the cumulative descriptor
                if( ry >= 0.0 )
                {
                    dxp += rx;
                    mdxp += fabs(rx);
                }
                else
                {
                    dxn += rx;
                    mdxn += fabs(rx);
                }

                if( rx >= 0.0 )
                {
                    dyp += ry;
                    mdyp += fabs(ry);
                }
                else
                {
                    dyn += ry;
                    mdyn += fabs(ry);
                }
            }
        }

        // Add the values to the descriptor vector
        gauss_s2 = gaussian(cx-2.0f,cy-2.0f,1.5f);

        kpt.descriptor[dcount++] = dxp*gauss_s2;
        kpt.descriptor[dcount++] = dxn*gauss_s2;
        kpt.descriptor[dcount++] = mdxp*gauss_s2;
        kpt.descriptor[dcount++] = mdxn*gauss_s2;
        kpt.descriptor[dcount++] = dyp*gauss_s2;
        kpt.descriptor[dcount++] = dyn*gauss_s2;
        kpt.descriptor[dcount++] = mdyp*gauss_s2;
        kpt.descriptor[dcount++] = mdyn*gauss_s2;

        // Store the current length^2 of the vector
        len += (dxp*dxp + dxn*dxn + mdxp*mdxp + mdxn*mdxn +
               dyp*dyp + dyn*dyn + mdyp*mdyp + mdyn*mdyn)*gauss_s2*gauss_s2;

        j += 9;
    }

        i += 9;
  }

  // convert to unit vector
  len = sqrt(len);

  for(int i = 0; i < dsize; i++)
  {
      kpt.descriptor[i] /= len;
  }

  if( USE_CLIPPING_NORMALIZATION == true )
  {
      Clipping_Descriptor(kpt,CLIPPING_NORMALIZATION_NITER,CLIPPING_NORMALIZATION_RATIO);
  }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the extended G-SURF descriptor of the provided keypoint
 * given the main orientation of the keypoint
 * @param kpt Input keypoint
 * @note Rectangular grid of 24 s x 24 s. Descriptor Length 128. The descriptor is inspired
 * from Agrawal et al., CenSurE: Center Surround Extremas for Realtime Feature Detection and Matching,
 * ECCV 2008
*/
void KAZE::Get_MSURF_Descriptor_128(Ipoint &kpt)
{
  float scale = 0.0, gauss_s1 = 0.0, gauss_s2 = 0.0;
  float rx = 0.0, ry = 0.0, rrx = 0.0, rry = 0.0, len = 0.0, xf = 0.0, yf = 0.0, ys = 0.0, xs = 0.0;
  float sample_x = 0.0, sample_y = 0.0, co = 0.0, si = 0.0, angle = 0.0;
  float fx = 0.0, fy = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
  float dxp = 0.0, dyp = 0.0, mdxp = 0.0, mdyp = 0.0;
  float dxn = 0.0, dyn = 0.0, mdxn = 0.0, mdyn = 0.0;
  int x1 = 0, y1 = 0, x2 = 0, y2 = 0, sample_step = 0, pattern_size = 0;
  int kx = 0, ky = 0, i = 0, j = 0, dcount = 0;
  int dsize = 0, level = 0;

  // Subregion centers for the 4x4 gaussian weighting
  float cx = -0.5, cy = 0.5;

  // Set the descriptor size and the sample and pattern sizes
  dsize = kpt.descriptor_size = 128;
  sample_step = 5;
  pattern_size = 12;

  // Get the information from the keypoint
  yf = kpt.yf;
  xf = kpt.xf;
  scale = kpt.scale;
  angle = kpt.angle;
  level = kpt.level;
  co = cos(angle);
  si = sin(angle);

  // Allocate the memory for the vector
  kpt.descriptor = vector<float>(kpt.descriptor_size);

  i = -8;

  // Calculate descriptor for this interest point
  // Area of size 24 s x 24 s
  while(i < pattern_size)
  {
     j = -8;
     i = i-4;

     cx += 1.0;
     cy = -0.5;

     while(j < pattern_size)
     {
        dxp=dxn=mdxp=mdxn=0.0;
        dyp=dyn=mdyp=mdyn=0.0;

        cy += 1.0f;
        j = j - 4;

        ky = i + sample_step;
        kx = j + sample_step;

        xs = xf + (-kx*scale*si + ky*scale*co);
        ys = yf + (kx*scale*co + ky*scale*si);

        for (int k = i; k < i + 9; ++k)
        {
            for (int l = j; l < j + 9; ++l)
            {
                // Get coords of sample point on the rotated axis
                sample_y = yf + (l*scale*co + k*scale*si);
                sample_x = xf + (-l*scale*si + k*scale*co);

                // Get the gaussian weighted x and y responses
                gauss_s1 = gaussian(xs-sample_x,ys-sample_y,2.5*scale);

                y1 = fRound(sample_y-.5);
                x1 = fRound(sample_x-.5);

                Check_Descriptor_Limits(x1,y1,img_width,img_height);

                y2 = fRound(sample_y+.5);
                x2 = fRound(sample_x+.5);

                Check_Descriptor_Limits(x2,y2,img_width,img_height);

                fx = sample_x-x1;
                fy = sample_y-y1;

                res1 = *(evolution[level].Lx.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Lx.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Lx.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Lx.ptr<float>(y2)+x2);
                rx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                res1 = *(evolution[level].Ly.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Ly.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Ly.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Ly.ptr<float>(y2)+x2);
                ry = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                // Get the x and y derivatives on the rotated axis
                rry = gauss_s1*(rx*co + ry*si);
                rrx = gauss_s1*(-rx*si + ry*co);

                // Sum the derivatives to the cumulative descriptor
                // Sum the derivatives to the cumulative descriptor
                if( rry >= 0.0 )
                {
                    dxp += rrx;
                    mdxp += fabs(rrx);
                }
                else
                {
                    dxn += rrx;
                    mdxn += fabs(rrx);
                }

                if( rrx >= 0.0 )
                {
                    dyp += rry;
                    mdyp += fabs(rry);
                }
                else
                {
                    dyn += rry;
                    mdyn += fabs(rry);
                }
            }
        }

        // Add the values to the descriptor vector
        gauss_s2 = gaussian(cx-2.0f,cy-2.0f,1.5f);

        kpt.descriptor[dcount++] = dxp*gauss_s2;
        kpt.descriptor[dcount++] = dxn*gauss_s2;
        kpt.descriptor[dcount++] = mdxp*gauss_s2;
        kpt.descriptor[dcount++] = mdxn*gauss_s2;
        kpt.descriptor[dcount++] = dyp*gauss_s2;
        kpt.descriptor[dcount++] = dyn*gauss_s2;
        kpt.descriptor[dcount++] = mdyp*gauss_s2;
        kpt.descriptor[dcount++] = mdyn*gauss_s2;

        // Store the current length^2 of the vector
        len += (dxp*dxp + dxn*dxn + mdxp*mdxp + mdxn*mdxn +
               dyp*dyp + dyn*dyn + mdyp*mdyp + mdyn*mdyn)*gauss_s2*gauss_s2;

        j += 9;
    }

        i += 9;
  }

  // convert to unit vector
  len = sqrt(len);

  for(int i = 0; i < dsize; i++)
  {
      kpt.descriptor[i] /= len;
  }

  if( USE_CLIPPING_NORMALIZATION == true )
  {
      Clipping_Descriptor(kpt,CLIPPING_NORMALIZATION_NITER,CLIPPING_NORMALIZATION_RATIO);
  }

}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the G-SURF upright extended descriptor
 * (no rotation invariant) of the provided keypoint
 * @param kpt Input keypoint
 * @note Rectangular grid of 20 s x 20 s. Descriptor Length 128. No additional
 * G-SURF descriptor as described in Pablo F. Alcantarilla, Luis M. Bergasa and
 * Andrew J. Davison, Gauge-SURF Descriptors, Image and Vision Computing 31(1), 2013
*/
void KAZE::Get_GSURF_Upright_Descriptor_128(Ipoint &kpt)
{
  float scale = 0.0, len = 0.0, xf = 0.0, yf = 0.0, sample_x = 0.0, sample_y = 0.0;
  float rx = 0.0, ry = 0.0, rxx = 0.0, rxy = 0.0, ryy = 0.0, modg = 0.0;
  float fx = 0.0, fy = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
  float dxp = 0.0, dyp = 0.0, mdxp = 0.0, mdyp = 0.0;
  float dxn = 0.0, dyn = 0.0, mdxn = 0.0, mdyn = 0.0, lvv = 0.0, lww = 0.0;
  int x1 = 0, y1 = 0, x2 = 0, y2 = 0, sample_step = 0, pattern_size = 0, dcount = 0;
  int dsize = 0, level = 0;

  // Set the descriptor size and the sample and pattern sizes
  dsize = kpt.descriptor_size = 128;
  sample_step = 5;
  pattern_size = 10;

  // Get the information from the keypoint
  yf = kpt.yf;
  xf = kpt.xf;
  scale = kpt.scale;
  level = kpt.level;

  // Allocate the memory for the vector
  kpt.descriptor = vector<float>(kpt.descriptor_size);

  // Calculate descriptor for this interest point
  for(int i = -pattern_size; i < pattern_size; i+=sample_step)
  {
    for(int j = -pattern_size; j < pattern_size; j+=sample_step)
    {
         dxp=dxn=mdxp=mdxn=0.0;
         dyp=dyn=mdyp=mdyn=0.0;

          for(float k = i; k < i + sample_step; k+=0.5)
          {
            for(float l = j; l < j + sample_step; l+=0.5)
            {
                sample_y = k*scale + yf;
                sample_x = l*scale + xf;

                y1 = (int)(sample_y-.5);
                x1 = (int)(sample_x-.5);

                Check_Descriptor_Limits(x1,y1,img_width,img_height);

                y2 = (int)(sample_y+.5);
                x2 = (int)(sample_x+.5);

                Check_Descriptor_Limits(x2,y2,img_width,img_height);

                fx = sample_x-x1;
                fy = sample_y-y1;

                res1 = *(evolution[level].Lx.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Lx.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Lx.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Lx.ptr<float>(y2)+x2);
                rx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                res1 = *(evolution[level].Ly.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Ly.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Ly.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Ly.ptr<float>(y2)+x2);
                ry = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                modg = pow(rx,2) + pow(ry,2);

                if( modg != 0.0 )
                {
                    res1 = *(evolution[level].Lxx.ptr<float>(y1)+x1);
                    res2 = *(evolution[level].Lxx.ptr<float>(y1)+x2);
                    res3 = *(evolution[level].Lxx.ptr<float>(y2)+x1);
                    res4 = *(evolution[level].Lxx.ptr<float>(y2)+x2);
                    rxx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                    res1 = *(evolution[level].Lxy.ptr<float>(y1)+x1);
                    res2 = *(evolution[level].Lxy.ptr<float>(y1)+x2);
                    res3 = *(evolution[level].Lxy.ptr<float>(y2)+x1);
                    res4 = *(evolution[level].Lxy.ptr<float>(y2)+x2);
                    rxy = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                    res1 = *(evolution[level].Lyy.ptr<float>(y1)+x1);
                    res2 = *(evolution[level].Lyy.ptr<float>(y1)+x2);
                    res3 = *(evolution[level].Lyy.ptr<float>(y2)+x1);
                    res4 = *(evolution[level].Lyy.ptr<float>(y2)+x2);
                    ryy = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                    // Lww = (Lx^2 * Lxx + 2*Lx*Lxy*Ly + Ly^2*Lyy) / (Lx^2 + Ly^2)
                    lww = (pow(rx,2)*rxx + 2.0*rx*rxy*ry + pow(ry,2)*ryy) / (modg);

                    // Lvv = (-2*Lx*Lxy*Ly + Lxx*Ly^2 + Lx^2*Lyy) / (Lx^2 + Ly^2)
                    lvv = (-2.0*rx*rxy*ry + rxx*pow(ry,2) + pow(rx,2)*ryy) /(modg);
                }
                else
                {
                      lww = 0.0;
                      lvv = 0.0;
                }

                // Sum the derivatives to the cumulative descriptor
                if( lww >= 0.0 )
                {
                    dxp += lvv;
                    mdxp += fabs(lvv);
                }
                else
                {
                    dxn += lvv;
                    mdxn += fabs(lvv);
                }

                if( lvv >= 0.0 )
                {
                    dyp += lww;
                    mdyp += fabs(lww);
                }
                else
                {
                    dyn += lww;
                    mdyn += fabs(lww);
                }
            }
          }

      // Add the values to the descriptor vector
      kpt.descriptor[dcount++] = dxp;
      kpt.descriptor[dcount++] = dxn;
      kpt.descriptor[dcount++] = mdxp;
      kpt.descriptor[dcount++] = mdxn;
      kpt.descriptor[dcount++] = dyp;
      kpt.descriptor[dcount++] = dyn;
      kpt.descriptor[dcount++] = mdyp;
      kpt.descriptor[dcount++] = mdyn;

      // Store the current length^2 of the vector
      len += dxp*dxp + dxn*dxn + mdxp*mdxp + mdxn*mdxn +
             dyp*dyp + dyn*dyn + mdyp*mdyp + mdyn*mdyn;
    }
  }

  // convert to unit vector
  len = sqrt(len);

  for(int i = 0; i < dsize; i++)
  {
      kpt.descriptor[i] /= len;
  }

  if( USE_CLIPPING_NORMALIZATION == true )
  {
      Clipping_Descriptor(kpt,CLIPPING_NORMALIZATION_NITER,CLIPPING_NORMALIZATION_RATIO);
  }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method computes the extended descriptor of the provided keypoint given the
 * main orientation
 * @param kpt Input keypoint
 * @note Rectangular grid of 20 s x 20 s. Descriptor Length 128. No additional
 * G-SURF descriptor as described in Pablo F. Alcantarilla, Luis M. Bergasa and
 * Andrew J. Davison, Gauge-SURF Descriptors, Image and Vision Computing 31(1), 2013
*/
void KAZE::Get_GSURF_Descriptor_128(Ipoint &kpt)
{
  float scale = 0.0, len = 0.0, xf = 0.0, yf = 0.0;
  float rx = 0.0, ry = 0.0, rxx = 0.0, rxy = 0.0, ryy = 0.0;
  float sample_x = 0.0, sample_y = 0.0, co = 0.0, si = 0.0, angle = 0.0;
  float fx = 0.0, fy = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
  float dxp = 0.0, dyp = 0.0, mdxp = 0.0, mdyp = 0.0;
  float dxn = 0.0, dyn = 0.0, mdxn = 0.0, mdyn = 0.0;
  float lvv = 0.0, lww = 0.0, modg = 0.0;
  int x1 = 0, y1 = 0, x2 = 0, y2 = 0, sample_step = 0, pattern_size = 0, dcount = 0;
  int dsize = 0, level = 0;

  // Set the descriptor size and the sample and pattern sizes
  dsize = kpt.descriptor_size = 128;
  sample_step = 5;
  pattern_size = 10;

  // Get the information from the keypoint
  yf = kpt.yf;
  xf = kpt.xf;
  scale = kpt.scale;
  angle = kpt.angle;
  level = kpt.level;
  co = cos(angle);
  si = sin(angle);

  // Allocate the memory for the vector
  kpt.descriptor = vector<float>(kpt.descriptor_size);

  // Calculate descriptor for this interest point
  for(int i = -pattern_size; i < pattern_size; i+=sample_step)
  {
    for(int j = -pattern_size; j < pattern_size; j+=sample_step)
    {
         dxp=dxn=mdxp=mdxn=0.0;
         dyp=dyn=mdyp=mdyn=0.0;

          for(float k = i; k < i + sample_step; k+=0.5)
          {
            for(float l = j; l < j + sample_step; l+=0.5)
            {
                // Get the coordinates of the sample point on the rotated axis
                sample_y = yf + (l*scale*co + k*scale*si);
                sample_x = xf + (-l*scale*si + k*scale*co);

                y1 = (int)(sample_y-.5);
                x1 = (int)(sample_x-.5);

                Check_Descriptor_Limits(x1,y1,img_width,img_height);

                y2 = (int)(sample_y+.5);
                x2 = (int)(sample_x+.5);

                Check_Descriptor_Limits(x2,y2,img_width,img_height);

                fx = sample_x-x1;
                fy = sample_y-y1;

                res1 = *(evolution[level].Lx.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Lx.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Lx.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Lx.ptr<float>(y2)+x2);
                rx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                res1 = *(evolution[level].Ly.ptr<float>(y1)+x1);
                res2 = *(evolution[level].Ly.ptr<float>(y1)+x2);
                res3 = *(evolution[level].Ly.ptr<float>(y2)+x1);
                res4 = *(evolution[level].Ly.ptr<float>(y2)+x2);
                ry = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                modg = pow(rx,2) + pow(ry,2);

                if( modg != 0.0 )
                {
                    res1 = *(evolution[level].Lxx.ptr<float>(y1)+x1);
                    res2 = *(evolution[level].Lxx.ptr<float>(y1)+x2);
                    res3 = *(evolution[level].Lxx.ptr<float>(y2)+x1);
                    res4 = *(evolution[level].Lxx.ptr<float>(y2)+x2);
                    rxx = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                    res1 = *(evolution[level].Lxy.ptr<float>(y1)+x1);
                    res2 = *(evolution[level].Lxy.ptr<float>(y1)+x2);
                    res3 = *(evolution[level].Lxy.ptr<float>(y2)+x1);
                    res4 = *(evolution[level].Lxy.ptr<float>(y2)+x2);
                    rxy = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                    res1 = *(evolution[level].Lyy.ptr<float>(y1)+x1);
                    res2 = *(evolution[level].Lyy.ptr<float>(y1)+x2);
                    res3 = *(evolution[level].Lyy.ptr<float>(y2)+x1);
                    res4 = *(evolution[level].Lyy.ptr<float>(y2)+x2);
                    ryy = (1.0-fx)*(1.0-fy)*res1 + fx*(1.0-fy)*res2 + (1.0-fx)*fy*res3 + fx*fy*res4;

                    // Lww = (Lx^2 * Lxx + 2*Lx*Lxy*Ly + Ly^2*Lyy) / (Lx^2 + Ly^2)
                    lww = (pow(rx,2)*rxx + 2.0*rx*rxy*ry + pow(ry,2)*ryy) / (modg);

                    // Lvv = (-2*Lx*Lxy*Ly + Lxx*Ly^2 + Lx^2*Lyy) / (Lx^2 + Ly^2)
                    lvv = (-2.0*rx*rxy*ry + rxx*pow(ry,2) + pow(rx,2)*ryy) /(modg);
                }
                else
                {
                      lww = 0.0;
                      lvv = 0.0;
                }

                // Sum the derivatives to the cumulative descriptor
                if( lww >= 0.0 )
                {
                    dxp += lvv;
                    mdxp += fabs(lvv);
                }
                else
                {
                    dxn += lvv;
                    mdxn += fabs(lvv);
                }

                if( lvv >= 0.0 )
                {
                    dyp += lww;
                    mdyp += fabs(lww);
                }
                else
                {
                    dyn += lww;
                    mdyn += fabs(lww);
                }
            }
        }

        // Add the values to the descriptor vector
        kpt.descriptor[dcount++] = dxp;
        kpt.descriptor[dcount++] = dxn;
        kpt.descriptor[dcount++] = mdxp;
        kpt.descriptor[dcount++] = mdxn;
        kpt.descriptor[dcount++] = dyp;
        kpt.descriptor[dcount++] = dyn;
        kpt.descriptor[dcount++] = mdyp;
        kpt.descriptor[dcount++] = mdyn;

        // Store the current length^2 of the vector
        len += dxp*dxp + dxn*dxn + mdxp*mdxp + mdxn*mdxn +
               dyp*dyp + dyn*dyn + mdyp*mdyp + mdyn*mdyn;
    }
  }

  // convert to unit vector
  len = sqrt(len);

  for(int i = 0; i < dsize; i++)
  {
      kpt.descriptor[i] /= len;
  }

  if( USE_CLIPPING_NORMALIZATION == true )
  {
      Clipping_Descriptor(kpt,CLIPPING_NORMALIZATION_NITER,CLIPPING_NORMALIZATION_RATIO);
  }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method performs a scalar non-linear diffusion step using AOS schemes
 * @param Ld Image at a given evolution step
 * @param Ldprev Image at a previous evolution step
 * @param c Conductivity image
 * @param stepsize Stepsize for the nonlinear diffusion evolution
 * @note If c is constant, the diffusion will be linear
 * If c is a matrix of the same size as Ld, the diffusion will be nonlinear
 * The stepsize can be arbitrarilly large
*/
void KAZE::AOS_Step_Scalar(cv::Mat &Ld, const cv::Mat &Ldprev, const cv::Mat &c, const float stepsize)
{
    //int64 t1 = cv::getTickCount(); cout << "Begin AOS schemes at " << t1 << endl;
    AOS_Rows(Ldprev,c,stepsize);
    AOS_Columns(Ldprev,c,stepsize);
    //int64 t2 = cv::getTickCount(); cout << "Finish AOS schemes. Exec-time: " << 1000.0*(t2-t1)/cv::getTickFrequency() << endl;

   Ld = 0.5*(Lty + Ltx.t());
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method performs a scalar non-linear diffusion step using AOS schemes
 * Diffusion in each dimension is computed independently in a different thread
 * @param Ld Image at a given evolution step
 * @param Ldprev Image at a previous evolution step
 * @param c Conductivity image
 * @param stepsize Stepsize for the nonlinear diffusion evolution
 * @note If c is constant, the diffusion will be linear
 * If c is a matrix of the same size as Ld, the diffusion will be nonlinear
 * The stepsize can be arbitrarilly large
*/
#if HAVE_BOOST_THREADING
void KAZE::AOS_Step_Scalar_Parallel(cv::Mat &Ld, const cv::Mat &Ldprev, const cv::Mat &c, const float stepsize)
{
    //int64 t1 = cv::getTickCount(); cout << "Begin AOS schemes at " << t1 << endl;
   boost::thread *AOSth1 = new boost::thread(&KAZE::AOS_Rows,this,Ldprev,c,stepsize);
   boost::thread *AOSth2 = new boost::thread(&KAZE::AOS_Columns,this,Ldprev,c,stepsize);
   
   AOSth1->join();
   AOSth2->join();
   //int64 t2 = cv::getTickCount(); cout << "Finish AOS schemes. Exec-time: " << 1000.0*(t2-t1)/cv::getTickFrequency() << endl;
   
   Ld = 0.5*(Lty + Ltx.t());

   delete AOSth1;
   delete AOSth2;
}
#endif

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method performs performs 1D-AOS for the image rows
 * @param Ldprev Image at a previous evolution step
 * @param c Conductivity image
 * @param stepsize Stepsize for the nonlinear diffusion evolution
*/
void KAZE::AOS_Rows(const cv::Mat &Ldprev, const cv::Mat &c, const float stepsize)
{
    //int64 t1 = cv::getTickCount(); cout << "Begin AOS_Rows at " << t1 << endl;
    // Operate on rows
    int qcols = qr.cols, qrows = qr.rows;
    if (qr.isContinuous() && c.isContinuous())
    {
        qcols *= qrows;
        qrows = 1;
    }
    for( int i = 0; i < qrows; i++ )
    {
        for( int j = 0; j < qcols; j++ )
        {
            *(qr.ptr<float>(i)+j) = *(c.ptr<float>(i)+j) + *(c.ptr<float>(i+1)+j);
        }
    }
   
   for( int j = 0; j < py.cols; j++ )
   {
       *(py.ptr<float>(0)+j) = *(qr.ptr<float>(0)+j);
       *(py.ptr<float>(py.rows-1)+j) = *(qr.ptr<float>(qr.rows-1)+j);
   }

   qcols = qr.cols, qrows = qr.rows;
   if (qr.isContinuous() && py.isContinuous())
   {
       qcols *= qrows-1;
       qrows = 1;
   }
   for( int i = 0; i < qrows; i++ )
   {
       for( int j = 0; j < qcols; j++ )
       {
           *(py.ptr<float>(i+1)+j) = *(qr.ptr<float>(i)+j) + *(qr.ptr<float>(i+1)+j);
       }
   }

   // a = 1 + t.*p; (p is -1*p)
   // b = -t.*q;
   ay = 1.0 + stepsize*py; // p is -1*p
   by = -stepsize*qr;

   // Call to Thomas algorithm now
   Thomas(ay,by,Ldprev,Lty);
   //int64 t2 = cv::getTickCount(); cout << "Finish AOS_Rows. Exec-time: " << 1000.0*(t2-t1)/cv::getTickFrequency() << endl;
   
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method performs performs 1D-AOS for the image columns
 * @param Ldprev Image at a previous evolution step
 * @param c Conductivity image
 * @param stepsize Stepsize for the nonlinear diffusion evolution
*/
void KAZE::AOS_Columns(const cv::Mat &Ldprev, const cv::Mat &c, const float stepsize)
{
    //int64 t1 = cv::getTickCount(); cout << "Begin AOS_Columns at " << t1 << endl;
    // Operate on columns
    int qcols = qc.cols, qrows = qc.rows;
    if (qc.isContinuous() && c.isContinuous())
    {
        qcols *= qrows;
        qrows = 1;
    }
    for( int i = 0; i < qrows; i++ )
    {
        for( int j = 0; j < qcols; j++ )
        {
            *(qc.ptr<float>(i)+j) = *(c.ptr<float>(i)+j) + *(c.ptr<float>(i)+j+1);
        }
    }

   for( int i = 0; i < px.rows; i++ )
   {
       *(px.ptr<float>(i)) = *(qc.ptr<float>(i));
       *(px.ptr<float>(i)+px.cols-1) = *(qc.ptr<float>(i)+qc.cols-1);
   }

   for( int j = 1; j < px.cols-1; j++ )
   {
       for( int i = 0; i < px.rows; i++ )
       {
           *(px.ptr<float>(i)+j) = *(qc.ptr<float>(i)+j-1) + *(qc.ptr<float>(i)+j);
       }
   }

   // a = 1 + t.*p';
   ax = 1.0 + stepsize*px.t();
   
   // b = -t.*q';
   bx = -stepsize*qc.t();
   
   // Call Thomas algorithm again
   // But take care since we need to transpose the solution!!
   Thomas(ax,bx,Ldprev.t(),Ltx);
   //int64 t2 = cv::getTickCount(); cout << "Finish AOS_Columns. Exec-time: " << 1000.0*(t2-t1)/cv::getTickFrequency() << endl;

}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method does the Thomas algorithm for solving a tridiagonal linear system
 * @note The matrix A must be strictly diagonally dominant for a stable solution
*/
void KAZE::Thomas(cv::Mat a, cv::Mat b, cv::Mat Ld, cv::Mat x)
{
   // Auxiliary variables
   int n = a.rows;
   cv::Mat m = cv::Mat::zeros(a.rows,a.cols,CV_32F);
   cv::Mat l = cv::Mat::zeros(b.rows,b.cols,CV_32F);
   cv::Mat y = cv::Mat::zeros(Ld.rows,Ld.cols,CV_32F);

   /** A*x = d;                                                                                  */
   /**    / a1 b1  0  0 0  ...    0 \  / x1 \ = / d1 \                                           */
   /**    | c1 a2 b2  0 0  ...    0 |  | x2 | = | d2 |                                           */
   /**    |  0 c2 a3 b3 0  ...    0 |  | x3 | = | d3 |                                           */
   /**    |  :  :  :  : 0  ...    0 |  |  : | = |  : |                                           */
   /**    |  :  :  :  : 0  cn-1  an |  | xn | = | dn |                                           */

   /** 1. LU decomposition
   / L = / 1                \        U = / m1 r1            \
   /     | l1 1             |            |    m2 r2          |
   /     |    l2 1          |            |        m3 r3      |
   /     |     : : :        |            |       :  :  :     |
   /     \           ln-1 1 /            \                mn /    */

   for( int j = 0; j < m.cols; j++ )
   {
       *(m.ptr<float>(0)+j) = *(a.ptr<float>(0)+j);
   }
   
   for( int j = 0; j < y.cols; j++ )
   {
       *(y.ptr<float>(0)+j) = *(Ld.ptr<float>(0)+j);
   }
   
   // 2. Forward substitution L*y = d for y
   for( int k = 1; k < n; k++ )
   {
       for( int j=0; j < l.cols; j++ )
       {
           *(l.ptr<float>(k-1)+j) = *(b.ptr<float>(k-1)+j) / *(m.ptr<float>(k-1)+j);
       }
       
       for( int j=0; j < m.cols; j++ )
       {
           *(m.ptr<float>(k)+j) = *(a.ptr<float>(k)+j) - *(l.ptr<float>(k-1)+j)*(*(b.ptr<float>(k-1)+j));
       }
       
       for( int j=0; j < y.cols; j++ )
       {
           *(y.ptr<float>(k)+j) = *(Ld.ptr<float>(k)+j) - *(l.ptr<float>(k-1)+j)*(*(y.ptr<float>(k-1)+j));
       }
   }
   
   // 3. Backward substitution U*x = y
   for( int j=0; j < y.cols; j++ )
   {
       *(x.ptr<float>(n-1)+j) = (*(y.ptr<float>(n-1)+j))/(*(m.ptr<float>(n-1)+j));
   }
   
   for( int i = n-2; i >= 0; i-- )
   {
       for( int j = 0; j < x.cols; j++ )
       {
           *(x.ptr<float>(i)+j) = (*(y.ptr<float>(i)+j) - (*(b.ptr<float>(i)+j))*(*(x.ptr<float>(i+1)+j)))/(*(m.ptr<float>(i)+j));
       }
   }
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method saves the nonlinear scale space into jpg images
*/
/*
void KAZE::Save_Nonlinear_Scale_Space(void)
{
   cv::Mat img_aux;
   char cad[NMAX_CHAR];

   for( unsigned int i = 0; i < evolution.size(); i++ )
   {
        Convert_Scale(evolution[i].Lt);
        evolution[i].Lt.convertTo(img_aux,CV_8U,255.0,0);
        sprintf(cad,"../../output/images\nl_evolution_%02d.jpg",i);
        cv::imwrite(cad,img_aux);
   }
}
*/
//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method saves the feature detector responses of the nonlinear scale space
 * into jpg images
*/
/*
void KAZE::Save_Detector_Responses(void)
{
   cv::Mat img_aux;
   char cad[NMAX_CHAR];
   
   for( unsigned int i = 0; i < evolution.size(); i++ )
   {
        Convert_Scale(evolution[i].Ldet);
        evolution[i].Ldet.convertTo(img_aux,CV_8U,255.0,0);
        sprintf(cad,"../../output/images\nl_detector_%02d.jpg",i);
        imwrite(cad,img_aux);
   }
}*/
//*************************************************************************************
//*************************************************************************************

/**
 * @brief This method saves the flow diffusivity responsesof the nonlinear scale space
 * into jpg images
*/
/*
void KAZE::Save_Flow_Responses(void)
{
   cv::Mat img_aux;
   char cad[NMAX_CHAR];
   
   for( unsigned int i = 0; i < evolution.size(); i++ )
   {
        Convert_Scale(evolution[i].Lflow);
        evolution[i].Lflow.convertTo(img_aux,CV_8U,255.0,0);
        sprintf(cad,"../../output/images/flow/flow_%02d.jpg",i);
        imwrite(cad,img_aux);
   }
}
*/
//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function computes the angle from the vector given by (X Y). From 0 to 2*Pi
*/
inline float Get_Angle(float X, float Y)
{
    
  if( X >= 0 && Y >= 0 )
  {
     return atan(Y/X);
  }
    
  if( X < 0 && Y >= 0 )
  {
     return PI - atan(-Y/X);
  }
      
  if( X < 0 && Y < 0 )
  {
     return PI + atan(Y/X); 
  }

  if( X >= 0 && Y < 0 )
  {
      return M2_PI - atan(-Y/X);
  }
    
  return 0;
}

//*************************************************************************************
//*************************************************************************************

/**
 * @brief This function performs descriptor clipping for a given keypoint
 * @param keypoint Input keypoint
 * @param iter Number of iterations
 * @param ratio Clipping ratio
*/
inline void Clipping_Descriptor(Ipoint &keypoint, int niter, float ratio)
{
    int dsize = keypoint.descriptor_size;
    float cratio = ratio / std::sqrtf(dsize);
    float len = 0.0;

    for( int i = 0; i < niter; i++ )
    {
        len = 0.0;
        for( int j = 0; j < dsize; j++ )
        {
            if( keypoint.descriptor[j] > cratio )
            {
                keypoint.descriptor[j] = cratio;
            }
            else if( keypoint.descriptor[j] < -cratio )
            {
                keypoint.descriptor[j] = -cratio;
            }
            len += keypoint.descriptor[j]*keypoint.descriptor[j];
        }
        
        // Normalize again
        len = sqrt(len);

        for( int j = 0; j < dsize; j++ )
        {
            keypoint.descriptor[j] = keypoint.descriptor[j] / len;
        }
    }
}

//**************************************************************************************
//**************************************************************************************

/**
 * @brief This function computes the value of a 2D Gaussian function
 * @param x X Position
 * @param y Y Position
 * @param sig Standard Deviation
*/
inline float gaussian(float x, float y, float sig)
{
  return exp(-(x*x+y*y)/(2.0f*sig*sig));
}

//**************************************************************************************
//**************************************************************************************

/**
 * @brief This function checks descriptor limits
 * @param x X Position
 * @param y Y Position
 * @param width Image width
 * @param height Image height
*/
inline void Check_Descriptor_Limits(int &x, int &y, int width, int height )
{
    if( x < 0 )
    {
        x = 0;
    }
                
    if( y < 0 )
    {
        y = 0;
    }
            
    if( x > width-1 )
    {
        x = width-1;
    }

    if( y > height-1 )
    {
        y = height-1;
    }
}
