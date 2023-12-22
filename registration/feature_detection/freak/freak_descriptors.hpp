#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <dirent.h>  
#include <unistd.h>
                                                                                                                                                                                                           
using namespace cv; 
using namespace std;

struct PatternPoint {
    float x; // x coordinate relative to center
    float y; // x coordinate relative to center
    float sigma; // Gaussian smoothing sigma
};

struct DescriptionPair {
    int i; // index of the first point
    int j; // index of the second point
};

struct OrientationPair {
    int i; // index of the first point
    int j; // index of the second point
    int weight_dx; // dx/(norm_sq))*4096
    int weight_dy; // dy/(norm_sq))*4096
};

struct PairStat { // used to sort pairs during pairs selection
    double mean;
    int idx;
};

struct sortMean {
    bool operator()( const PairStat& a, const PairStat& b ) const {
        return a.mean < b.mean;
    }
};

const double SQRT2 = 1.4142135623731;
const double INV_SQRT2 = 1.0 / SQRT2;
const double LOG2 = 0.693147180559945;
const int NB_SCALES = 1;
const int NB_ORIENTATION = 256;
const int NB_POINTS = 43;
const int NB_PAIRS = 512;
const int SMALLEST_KP_SIZE = 7; // smallest size of keypoints
const int NB_ORIENPAIRS = 45;
const int DEF_PAIRS[NB_PAIRS] = { // default pairs
                                    404,431,818,511,181,52,311,874,774,543,719,230,417,205,11,
                                    560,149,265,39,306,165,857,250,8,61,15,55,717,44,412,
                                    592,134,761,695,660,782,625,487,549,516,271,665,762,392,178,
                                    796,773,31,672,845,548,794,677,654,241,831,225,238,849,83,
                                    691,484,826,707,122,517,583,731,328,339,571,475,394,472,580,
                                    381,137,93,380,327,619,729,808,218,213,459,141,806,341,95,
                                    382,568,124,750,193,749,706,843,79,199,317,329,768,198,100,
                                    466,613,78,562,783,689,136,838,94,142,164,679,219,419,366,
                                    418,423,77,89,523,259,683,312,555,20,470,684,123,458,453,833,
                                    72,113,253,108,313,25,153,648,411,607,618,128,305,232,301,84,
                                    56,264,371,46,407,360,38,99,176,710,114,578,66,372,653,
                                    129,359,424,159,821,10,323,393,5,340,891,9,790,47,0,175,346,
                                    236,26,172,147,574,561,32,294,429,724,755,398,787,288,299,
                                    769,565,767,722,757,224,465,723,498,467,235,127,802,446,233,
                                    544,482,800,318,16,532,801,441,554,173,60,530,713,469,30,
                                    212,630,899,170,266,799,88,49,512,399,23,500,107,524,90,
                                    194,143,135,192,206,345,148,71,119,101,563,870,158,254,214,
                                    276,464,332,725,188,385,24,476,40,231,620,171,258,67,109,
                                    844,244,187,388,701,690,50,7,850,479,48,522,22,154,12,659,
                                    736,655,577,737,830,811,174,21,237,335,353,234,53,270,62,
                                    182,45,177,245,812,673,355,556,612,166,204,54,248,365,226,
                                    242,452,700,685,573,14,842,481,468,781,564,416,179,405,35,
                                    819,608,624,367,98,643,448,2,460,676,440,240,130,146,184,
                                    185,430,65,807,377,82,121,708,239,310,138,596,730,575,477,
                                    851,797,247,27,85,586,307,779,326,494,856,324,827,96,748,
                                    13,397,125,688,702,92,293,716,277,140,112,4,80,855,839,1,
                                    413,347,584,493,289,696,19,751,379,76,73,115,6,590,183,734,
                                    197,483,217,344,330,400,186,243,587,220,780,200,793,246,824,
                                    41,735,579,81,703,322,760,720,139,480,490,91,814,813,163,
                                    152,488,763,263,425,410,576,120,319,668,150,160,302,491,515,
                                    260,145,428,97,251,395,272,252,18,106,358,854,485,144,550,
                                    131,133,378,68,102,104,58,361,275,209,697,582,338,742,589,
                                    325,408,229,28,304,191,189,110,126,486,211,547,533,70,215,
                                    670,249,36,581,389,605,331,518,442,822
                                   };

class MyFreakDescriptorsTest{
    public:
        MyFreakDescriptorsTest();
        ~MyFreakDescriptorsTest();

        Mat run(Mat src, vector<Point>);

    private:
        void BuildPattern();
        int MeanIntensity(Mat image, Mat integral, float kp_x, float kp_y,
                        int scale, int rot, int point);
        Mat ComputeDescriptors(Mat image, vector<Point> keypoints);
        void extractDescriptor(int *pointsValue, void ** ptr);
        Mat SelectPairs(Mat image, vector<Point>& key_points, double corrTresh);

    private:
        bool orientationNormalized; //true if the orientation is normalized, false otherwise
        bool scaleNormalized; //true if the scale is normalized, false otherwise
        float patternScale; //scaling of the pattern
        int nOctaves;
        bool extAll = false; // true if all pairs need to be extracted for pairs selection
        PatternPoint* patternLookup; // look-up table for the pattern points (position+sigma of all points at all scales and orientation)
        int patternSizes[NB_SCALES]; // size of the pattern at a specific scale (used to check if a point is within image boundaries)
        DescriptionPair descriptionPairs[NB_PAIRS];
        OrientationPair orientationPairs[NB_ORIENPAIRS];
        int max_border, max_sigma;
};
