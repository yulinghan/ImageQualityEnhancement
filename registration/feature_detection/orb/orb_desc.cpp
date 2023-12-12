#include "orb_desc.hpp"

MyOrbDescTest::MyOrbDescTest() {
}

MyOrbDescTest::~MyOrbDescTest() {
}

float MyOrbDescTest::get_value(int idx, vector<Point> pattern, float a, float b, Mat src, int cx, int cy) {
	float x = pattern[idx].x * a - pattern[idx].y * b;
	float y = pattern[idx].x * b + pattern[idx].y * a;
	int ix = int(round(x));
	int iy = int(round(y));
	return src.at<uchar>(cy + iy, cx + ix);
}

vector<vector<int>> MyOrbDescTest::computeOrbDescriptors(vector<Mat> src_arr, vector<KeyPoint> keypoints, vector<Point> pattern, int border) {
	vector<vector<int>> descriptors;
    int dsize = 16;

	for(int i=0; i<keypoints.size(); i++) {
		KeyPoint kpt = keypoints[i];
		int layer = kpt.octave;
		float scale = 1.0 / pow(2, kpt.octave);
		float angle = kpt.angle / 180.0 * M_PI;
		float a = cos(angle), b = sin(angle);

		int cx = round(kpt.pt.x * scale) + border;
        int cy = round(kpt.pt.y * scale) + border;
		vector<int> des;
		int pattern_idx = 0;

		for(int j=0; j<dsize; j++) {
			int byte_v = 0;
			for(int nn=0; nn<8; nn++) {
				float t0 = get_value(pattern_idx+2*nn, pattern, a, b, src_arr[layer], cx, cy);
				float t1 = get_value(pattern_idx+2*nn+1, pattern, a, b, src_arr[layer], cx, cy);
				int bit_v = int(t0 < t1);
				byte_v += (bit_v << nn);
			}
			des.push_back(byte_v);
			pattern_idx += dsize;
		}
		descriptors.push_back(des);
	}

	return descriptors;
}

vector<vector<int>> MyOrbDescTest::Run(vector<Mat> src_arr, vector<KeyPoint> keypoints, int border) {
    for(int level=0; level<src_arr.size(); level++) {
        GaussianBlur(src_arr[level], src_arr[level], Size(7, 7), 2, 2, BORDER_REFLECT_101);
    }

	int nkeypoints = keypoints.size();
	int npoints = 512;

	vector<Point> pattern;
	for (size_t i = 0; i < npoints; i++) {
		pattern.push_back(Point(bit_pattern_31_[2 * i], bit_pattern_31_[2 * i + 1]));
    }

    vector<vector<int>> descriptors = computeOrbDescriptors(src_arr, keypoints, pattern, border);

    return descriptors;
}
