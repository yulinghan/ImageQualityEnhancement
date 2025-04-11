#include "pchip.hpp"

CPchip::CPchip(vector<float> X, vector<float> Y, int n, double x) {
	int k = FindIndex(X, n, x);
    vector<float> h, delta;

	for (int i = 0; i <= n - 2; i++) {
		h.push_back(X[i + 1] - X[i]);
		if (h[i] < 1e-12) {
			abort();
		}
		delta.push_back((Y[i + 1] - Y[i]) / h[i]);
	}
	double diff1 = ComputeDiff(h, delta, n, k);
	double diff2 = ComputeDiff(h, delta, n, k + 1);

	double a = (diff2 + diff1 - 2 * delta[k]) / h[k] / h[k];
	double b = (-diff2 - 2 * diff1 + 3 * delta[k]) / h[k];
	double c = diff1;
	double d = Y[k];
	y = a * pow(x - X[k], 3) + b * pow(x - X[k], 2) + c * (x - X[k]) + d;
}

CPchip::~CPchip(){
}

int CPchip::FindIndex(vector<float> X, int n, double x)
{
	/* 找到插值位置 */
	if (n < 3) {
		abort();
	}
	int index = 0;
	int iBeg = 0;
	if (x <= X[1]) {
		index = 0;
	} else if (x >= X[n - 2]) {
		index = n - 2;
	} else {
		iBeg = floor((n - 1) / (X[n - 1] - X[0])*(x - X[0]));
		if (x >= X[iBeg]) {
			for (int i = iBeg; i <= n - 3; i++) {
				if (x >= X[i] && x <= X[i + 1]) {
					index = i;
					break;
				}
			}
		} else {
			for (int i = iBeg - 1; i >= 1; i--) {
				if (x >= X[i] && x <= X[i + 1]) {
					index = i;
					break;
				}
			}
		}
	}
	return index;
}

double CPchip::ComputeDiff(vector<float> h, vector<float> delta, int n, int k)
{
	double diff = 0;
	if (k == 0) {
		double t = ((2 * h[0] + h[1])*delta[0] - h[0] * delta[1]) / (h[0] + h[1]);
		if (t*delta[0] <= 0)
			diff = 0;
		else if (delta[0] * delta[1] < 0 && abs(t) > abs(3 * delta[0]))
			diff = 3 * delta[0];
		else
			diff = t;
	} else if (k == n - 1) {
		double t = ((2 * h[n - 2] + h[n - 3])*delta[n - 2] - h[n - 2] * delta[n - 3]) / (h[n - 2] + h[n - 3]);
		if (t*delta[n - 2] <= 0)
			diff = 0;
		else if (delta[n - 2] * delta[n - 3] < 0 && abs(t) > abs(3 * delta[n - 2]))
			diff = 3 * delta[n - 2];
		else
			diff = t;
	} else {
		if (delta[k] * delta[k - 1] <= 0)
			diff = 0;
		else if (delta[k] * delta[k - 1] > 0 && abs(h[k] - h[k - 1]) < 1e-12)
			diff = 2 * delta[k] * delta[k - 1] / (delta[k] + delta[k - 1]);
		else {
			double w1 = 2 * h[k] + h[k - 1];
			double w2 = h[k] + 2 * h[k - 1];
			diff = delta[k] * delta[k - 1] / (w1*delta[k] + w2 * delta[k - 1])*(w1 + w2);
		}
	}
	return diff;
}
