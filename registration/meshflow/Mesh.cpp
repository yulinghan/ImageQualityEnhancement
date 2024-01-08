#include "Mesh.h"

Quad::Quad()
{
	V00.x = 0.0; V00.y = 0.0;
	V01.x = 0.0; V01.y = 0.0;
	V10.x = 0.0; V10.y = 0.0;
	V11.x = 0.0; V11.y = 0.0;
}

Quad::Quad(const Quad &inQuad)
{
	V00.x = inQuad.V00.x; V00.y = inQuad.V00.y;
	V01.x = inQuad.V01.x; V01.y = inQuad.V01.y;
	V10.x = inQuad.V10.x; V10.y = inQuad.V10.y;
	V11.x = inQuad.V11.x; V11.y = inQuad.V11.y;
}

Quad::Quad(const cv::Point2f &inV00, const cv::Point2f &inV01, const cv::Point2f &inV10, const cv::Point2f &inV11)
{
	V00.x = inV00.x; V00.y = inV00.y;
	V01.x = inV01.x; V01.y = inV01.y;
	V10.x = inV10.x; V10.y = inV10.y;
	V11.x = inV11.x; V11.y = inV11.y;
}

Quad::~Quad()
{

}

void Quad::operator=(const Quad &inQuad)
{
	V00.x = inQuad.V00.x; V00.y = inQuad.V00.y;
	V01.x = inQuad.V01.x; V01.y = inQuad.V01.y;
	V10.x = inQuad.V10.x; V10.y = inQuad.V10.y;
	V11.x = inQuad.V11.x; V11.y = inQuad.V11.y;
}

double Quad::getMinX() const
{
	float minx = min(V00.x, V01.x);
	minx = min(minx, V10.x);
	minx = min(minx, V11.x);

	return 1.0*minx;
}

double Quad::getMaxX() const
{
	float maxX = max(V00.x, V01.x);
	maxX = max(maxX, V10.x);
	maxX = max(maxX, V11.x);

	return 1.0*maxX;
}

double Quad::getMinY() const
{
	float minY = min(V00.y, V01.y);
	minY = min(minY, V10.y);
	minY = min(minY, V11.y);

	return 1.0*minY;
}

double Quad::getMaxY() const
{
	float maxY = max(V00.y, V01.y);
	maxY = max(maxY, V10.y);
	maxY = max(maxY, V11.y);

	return 1.0*maxY;
}

bool Quad::isPointIn(const cv::Point2f &pt) const
{
	bool in1 = isPointInTriangular(pt, V00, V01, V11);
	bool in2 = isPointInTriangular(pt, V00, V10, V11);
	if (in1 || in2)
	{
		return true;
	}
	else
	{
		return false;
	}
}

bool Quad::getBilinearCoordinates(const cv::Point2f &pt, vector<double> &coefficient) const
{
	double k1 = -1;
	double k2 = -1;

	double a_x = V00.x - V01.x - V10.x + V11.x;
	double b_x = -V00.x + V01.x;
	double c_x = -V00.x + V10.x;
	double d_x = V00.x - pt.x;

	double a_y = V00.y - V01.y - V10.y + V11.y;
	double b_y = -V00.y + V01.y;
	double c_y = -V00.y + V10.y;
	double d_y = V00.y - pt.y;

	double bigA = -a_y*b_x + b_y*a_x;
	double bigB = -a_y*d_x - c_y*b_x + d_y*a_x + b_y*c_x;
	double bigC = -c_y*d_x + d_y*c_x;

	double tmp1 = -1;
	double tmp2 = -1;
	double tmp3 = -1;
	double tmp4 = -1;

	if (bigB*bigB - 4 * bigA*bigC >= 0.0)
	{
		if (abs(bigA) >= 0.000001)
		{
			tmp1 = (-bigB + sqrt(bigB*bigB - 4 * bigA*bigC)) / (2 * bigA);
			tmp2 = (-bigB - sqrt(bigB*bigB - 4 * bigA*bigC)) / (2 * bigA);
		}
		else
		{
			tmp1 = -bigC / bigB;
		}

		if (tmp1 >= -0.999999 && tmp1 <= 1.000001)
		{
			// 			if (abs(a_x*tmp1 + c_x) < 0.000001)
			// 			{
			// 				tmp3 = -(b_y*tmp1 + d_y) / (a_y*tmp1 + c_y);
			// 			}
			// 			else
			// 			{
			// 				tmp3 = -(b_x*tmp1 + d_x) / (a_x*tmp1 + c_x);
			// 			}
			/*			printf("%f\n", tmp3);*/

			tmp3 = -(b_y*tmp1 + d_y) / (a_y*tmp1 + c_y);
			tmp4 = -(b_x*tmp1 + d_x) / (a_x*tmp1 + c_x);
			if (tmp3 >= -0.999999 && tmp3 <= 1.000001)
			{
				k1 = tmp1;
				k2 = tmp3;
			}
			else if (tmp4 >= -0.999999 && tmp4 <= 1.000001)
			{
				k1 = tmp1;
				k2 = tmp4;
			}
		}
		if (tmp2 >= -0.999999 && tmp2 <= 1.000001)
		{
			// 			if (abs(a_x*tmp1 + c_x) < 0.000001)
			// 			{
			// 				tmp3 = -(b_y*tmp2 + d_y) / (a_y*tmp2 + c_y);
			// 			}
			// 			else
			// 			{
			// 				tmp3 = -(b_x*tmp2 + d_x) / (a_x*tmp2 + c_x);
			// 			}
			tmp3 = -(b_y*tmp2 + d_y) / (a_y*tmp2 + c_y);
			tmp4 = -(b_x*tmp2 + d_x) / (a_x*tmp2 + c_x);
			if (tmp3 >= -0.999999 && tmp3 <= 1.000001)
			{
				k1 = tmp2;
				k2 = tmp3;
			}
			else if (tmp4 >= -0.999999 && tmp4 <= 1.000001)
			{
				k1 = tmp2;
				k2 = tmp4;
			}
		}
	}
	if (k1 >= -0.999999 && k1 <= 1.000001 && k2 >= -0.999999 && k2 <= 1.000001)
	{
		double coe1 = (1.0 - k1)*(1.0 - k2);
		double coe2 = k1*(1.0 - k2);
		double coe3 = (1.0 - k1)*k2;
		double coe4 = k1*k2;

		coefficient.push_back(coe1);
		coefficient.push_back(coe2);
		coefficient.push_back(coe3);
		coefficient.push_back(coe4);

		return true;
	}
	else
	{
		// 		printf("(%lf, %lf), (%lf, %lf), (%lf, %lf), (%lf, %lf) -> (%lf, %lf) -> (%lf, %lf)\n", V00.x, V00.y, V01.x, V01.y, V10.x, V10.y, V11.x, V11.y, pt.x, pt.y, k1, k2);
		// 		printf("(%lf, %lf, %lf, %lf) -> (%lf, %lf, %lf, %lf)\n", a_x, b_x, c_x, d_x, a_y, b_y, c_y, d_x);
		// 		printf("(%lf, %lf, %lf, %lf) -> (%lf, %lf, %lf)\n", tmp1, tmp2, tmp3, tmp4, bigA, bigB, bigC);
		return false;
	}


	// 	if (V01.x-V00.x != 0.0 && V11.x-V10.x!= 0.0)
	// 	{
	// 		double k1 = (pt.x-V00.x)/(V01.x-V00.x);
	// 		double k2 = (pt.x-V10.x)/(V11.x-V10.x);
	// 		double B0y = (1-k1)*V00.y+k1*V01.y;
	// 		double B1y = (1-k2)*V10.y+k2*V11.y;
	// 		if (B1y-B0y != 0.0)
	// 		{
	// 			double k3 = (pt.y-B0y)/(B1y-B0y);
	// 			double c00 = (1-k1)*(1-k3);
	// 			double c01 = k1*(1-k3);
	// 			double c10 = (1-k2)*k3;
	// 			double c11 = k2*k3;
	// 			coefficient.push_back(c00);
	// 			coefficient.push_back(c01);
	// 			coefficient.push_back(c10);
	// 			coefficient.push_back(c11);
	// 
	// 			return true;
	// 		}
	// 		else
	// 		{
	// 			return false;
	// 		}
	// 	}
	// 	else
	// 	{
	// 		return false;
	// 	}
}

bool Quad::getBilinearCoordinates(const cv::Point2f &pt, double* &coefficient) const{

	if (!this->isPointIn(pt)) 	return false;

	if (pt.x == 0 && pt.y == 0)      return false;

	double d1, d2, d3, d4;

	double x0, y0, x1, y1, x, y;

	x0 = this->V00.x;
	y0 = this->V00.y;
	x1 = this->V11.x;
	y1 = this->V11.y;
	x = pt.x;
	y = pt.y;

	d1 = y - y0; d1 = abs(d1);
	d2 = y - y1; d2 = abs(d2);
	d3 = x - x0; d3 = abs(d3);
	d4 = x - x1; d4 = abs(d4);

	coefficient[0] = d2*d4 / (d1*d3 + d2*d3 + d1*d4 + d2*d4);//V00
	coefficient[1] = d1*d4 / (d1*d3 + d2*d3 + d1*d4 + d2*d4);//V01
	coefficient[2] = d2*d3 / (d1*d3 + d2*d3 + d1*d4 + d2*d4);//V10
	coefficient[3] = d1*d3 / (d1*d3 + d2*d3 + d1*d4 + d2*d4);//V11

	return true;
}

cv::Point2f Quad::getPointByBilinearCoordinates(const vector<double> &coefficient) const
{
	double resx = V00.x*coefficient[0] + V01.x*coefficient[1] + V10.x*coefficient[2] + V11.x*coefficient[3];
	double resy = V00.y*coefficient[0] + V01.y*coefficient[1] + V10.y*coefficient[2] + V11.y*coefficient[3];

	return cv::Point2f(resx, resy);
}


//////////////////////////////////////////////////////////////////////////////////////////////

Mesh::Mesh()
{
	imgRows = 0;
	imgCols = 0;
	width = 0;
	height = 0;
}

Mesh::Mesh(const Mesh &inMesh)
{
	imgRows = inMesh.imgRows;
	imgCols = inMesh.imgCols;
	quadHeight = inMesh.quadHeight;
	quadWidth = inMesh.quadWidth;
	width = inMesh.width;
	height = inMesh.height;
	xMat = cv::Mat::zeros(height, width, CV_64FC1);
	yMat = cv::Mat::zeros(height, width, CV_64FC1);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			setVertex(i, j, inMesh.getVertex(i, j));
		}
	}
}

Mesh::Mesh(int rows, int cols)
{
	imgRows = rows;
	imgCols = cols;
	width = 0;
	height = 0;
}

Mesh::Mesh(int rows, int cols, double quadWidth, double quadHeight)
{
	imgRows = rows;
	imgCols = cols;
	buildMesh(quadWidth, quadHeight);
	this->quadWidth = quadWidth;
	this->quadHeight = quadHeight;
}

Mesh::~Mesh()
{
}

void Mesh::operator=(const Mesh &inMesh)
{
	imgRows = inMesh.imgRows;
	imgCols = inMesh.imgCols;
	width = inMesh.width;
	height = inMesh.height;
	quadHeight = inMesh.quadHeight;
	quadWidth = inMesh.quadWidth;
	xMat = cv::Mat::zeros(height, width, CV_64FC1);
	yMat = cv::Mat::zeros(height, width, CV_64FC1);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			setVertex(i, j, inMesh.getVertex(i, j));
		}
	}
}

double Mesh::differentFrom(const Mesh &inMesh) const
{
	double diff = 0.0;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			cv::Point2f tmp1 = getVertex(i, j);
			cv::Point2f tmp2 = inMesh.getVertex(i, j);
			diff += (tmp1.x - tmp2.x)*(tmp1.x - tmp2.x) + (tmp1.y - tmp2.y)*(tmp1.y - tmp2.y);
			if (diff > 0.0)
			{
				printf("%.12f\n", diff);
			}
		}
	}
	return diff;
}

cv::Point2f Mesh::getVertex(int i, int j) const
{
	double x;
	double y;

	x = xMat.at<double>(i, j);
	y = yMat.at<double>(i, j);

	return cv::Point2f(x, y);
}

Quad Mesh::getQuad(int i, int j) const
{
	cv::Point2f V00;
	cv::Point2f V01;
	cv::Point2f V10;
	cv::Point2f V11;

	V00 = getVertex(i - 1, j - 1);
	V01 = getVertex(i - 1, j);
	V10 = getVertex(i, j - 1);
	V11 = getVertex(i, j);

	Quad qd(V00, V01, V10, V11);

	return qd;
}

void Mesh::setVertex(int i, int j, const cv::Point2f &pos)
{
	xMat.at<double>(i, j) = pos.x;
	yMat.at<double>(i, j) = pos.y;
}

void Mesh::initialize(int w, int h)
{
	width = w;
	height = h;
	xMat = cv::Mat::zeros(height, width, CV_64FC1) - 10000;
	yMat = cv::Mat::zeros(height, width, CV_64FC1) - 10000;
}

void Mesh::buildMesh(double quadWidth, double quadHeight)
{
	vector<int> xSet;
	vector<int> ySet;

	for (int x = 0; imgCols - x > 0.5*quadWidth; x += quadWidth)
	{
		xSet.push_back(x);
	}
	xSet.push_back(imgCols - 1);
	for (int y = 0; imgRows - y > 0.5*quadHeight; y += quadHeight)
	{
		ySet.push_back(y);
	}
	ySet.push_back(imgRows - 1);

	width = xSet.size();
	height = ySet.size();

	xMat.create(height, width, CV_64FC1);
	yMat.create(height, width, CV_64FC1);

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			xMat.at<double>(y, x) = xSet[x];
			yMat.at<double>(y, x) = ySet[y];
		}
	}
}

void Mesh::drawMesh(cv::Mat &targetImg){

	cv::Mat temp = targetImg.clone();
	//cv::Scalar color(0,0,0);
	cv::Scalar color(255, 255, 255);
	int gap = 0;
	int lineWidth = 3;

	for (int i = 1; i < height; i++)
	{
		for (int j = 1; j < width; j++)
		{
			cv::Point2f pUp = getVertex(i - 1, j);
			cv::Point2f pLeft = getVertex(i, j - 1);
			cv::Point2f pCur = getVertex(i, j);

			pUp.x += gap;
			pUp.y += gap;
			pLeft.x += gap;
			pLeft.y += gap;
			pCur.x += gap;
			pCur.y += gap;

			if (pUp.x > -9999.0 && pUp.y > -9999.0 && pCur.x > -9999.0 && pCur.y > -9999.0){
				double dis = sqrt((pUp.x - pCur.x)*(pUp.x - pCur.x) + (pUp.y - pCur.y)*(pUp.y - pCur.y));
				//if(dis<100){
				line(temp, cv::Point2f(pUp.x, pUp.y), cv::Point2f(pCur.x, pCur.y), color, lineWidth, cv::LINE_AA);
				//}
			}
			if (pLeft.x > -9999.0 && pLeft.y > -9999.0 && pCur.x > -9999.0 && pCur.y > -9999.0){
				double dis = sqrt((pLeft.x - pCur.x)*(pLeft.x - pCur.x) + (pLeft.y - pCur.y)*(pLeft.y - pCur.y));
				//if(dis<100){
				line(temp, cv::Point2f(pLeft.x, pLeft.y), cv::Point2f(pCur.x, pCur.y), color, lineWidth, cv::LINE_AA);
				//}
			}
			cv::circle(temp, cv::Point(pUp.x, pUp.y), lineWidth + 2, cv::Scalar(45, 57, 167), -1);
			cv::circle(temp, cv::Point(pLeft.x, pLeft.y), lineWidth + 2, cv::Scalar(45, 57, 167), -1);
			cv::circle(temp, cv::Point(pCur.x, pCur.y), lineWidth + 2, cv::Scalar(45, 57, 167), -1);
		}
	}

	for (int i = 1; i < height; i++)
	{
		cv::Point2f pLeft = getVertex(i, 0);
		cv::Point2f pLeftUp = getVertex(i - 1, 0);

		pLeftUp.x += gap;
		pLeftUp.y += gap;
		pLeft.x += gap;
		pLeft.y += gap;

		if (pLeft.x > -9999.0 && pLeft.y > -9999.0 && pLeftUp.x > -9999.0 && pLeftUp.y > -9999.0){
			double dis = sqrt((pLeft.x - pLeftUp.x)*(pLeft.x - pLeftUp.x) + (pLeft.y - pLeftUp.y)*(pLeft.y - pLeftUp.y));
			//if(dis<100){
			line(temp, cv::Point2f(pLeft.x, pLeft.y), cv::Point2f(pLeftUp.x, pLeftUp.y), color, lineWidth, cv::LINE_AA);
			//}
		}
		cv::circle(temp, cv::Point(pLeftUp.x, pLeftUp.y), lineWidth + 2, cv::Scalar(45, 57, 167), -1);
		cv::circle(temp, cv::Point(pLeft.x, pLeft.y), lineWidth + 2, cv::Scalar(45, 57, 167), -1);
	}

	for (int j = 1; j < width; j++)
	{
		cv::Point2f pLeftUp = getVertex(0, j - 1);
		cv::Point2f pUp = getVertex(0, j);

		pLeftUp.x += gap;
		pLeftUp.y += gap;
		pUp.x += gap;
		pUp.y += gap;

		if (pLeftUp.x > -9999.0 && pLeftUp.y > -9999.0 && pUp.x > -9999.0 && pUp.y > -9999.0){
			double dis = sqrt((pLeftUp.x - pUp.x)*(pLeftUp.x - pUp.x) + (pLeftUp.y - pUp.y)*(pLeftUp.y - pUp.y));
			//if(dis<100){
			line(temp, cv::Point2f(pLeftUp.x, pLeftUp.y), cv::Point2f(pUp.x, pUp.y), color, lineWidth, cv::LINE_AA);
			//}
		}
		cv::circle(temp, cv::Point(pUp.x, pUp.y), lineWidth + 2, cv::Scalar(45, 57, 167), -1);
		cv::circle(temp, cv::Point(pLeftUp.x, pLeftUp.y), lineWidth + 2, cv::Scalar(45, 57, 167), -1);
	}
	targetImg = (2.0 / 5 * targetImg + 3.0 / 5 * temp);
}

bool Mesh::selfCheck()
{
	bool res = true;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			cv::Point2f curV = getVertex(i, j);
			if (i > 0)
			{
				cv::Point2f upV = getVertex(i - 1, j);
				if (curV.y - upV.y < 2.0)
				{
					cv::Point2f newCurV(curV.x, (curV.y + upV.y) / 2 + 1);
					cv::Point2f newUpV(upV.x, (curV.y + upV.y) / 2 - 1);
					setVertex(i, j, newCurV);
					setVertex(i - 1, j, newUpV);
					for (int t = i - 1; t > 0; t--)
					{
						cv::Point2f tmpCur = getVertex(t, j);
						cv::Point2f tmpUp = getVertex(t - 1, j);
						if (tmpCur.y - tmpUp.y < 2.0)
						{
							tmpUp.y = tmpCur.y - 2.0;
							setVertex(t - 1, j, tmpUp);
						}
						else
						{
							break;
						}
					}
					for (int t = i; t < height; t++)
					{
						cv::Point2f tmpCur = getVertex(t, j);
						cv::Point2f tmpDown = getVertex(t + 1, j);
						if (tmpDown.y - tmpCur.y < 2.0)
						{
							tmpDown.y = tmpCur.y + 2.0;
							setVertex(t + 1, j, tmpDown);
						}
						else
						{
							break;
						}
					}
					res = false;
				}
			}
			if (j > 0)
			{
				cv::Point2f leftV = getVertex(i, j - 1);
				if (curV.x - leftV.x < 2.0)
				{
					cv::Point2f newCurV((curV.x + leftV.x) / 2 + 1, curV.y);
					cv::Point2f newLeftV((curV.x + leftV.x) / 2 - 1, leftV.y);
					setVertex(i, j, newCurV);
					setVertex(i, j - 1, newLeftV);
					for (int t = j - 1; t > 0; t--)
					{
						cv::Point2f tmpCur = getVertex(i, t);
						cv::Point2f tmpLeft = getVertex(i, t - 1);
						if (tmpCur.x - tmpLeft.x < 2.0)
						{
							tmpLeft.x = tmpCur.x - 2.0;
							setVertex(i, t - 1, tmpLeft);
						}
						else
						{
							break;
						}
					}
					for (int t = j; t < width; t++)
					{
						cv::Point2f tmpCur = getVertex(i, t);
						cv::Point2f tmpRight = getVertex(i, t + 1);
						if (tmpRight.x - tmpCur.x < 2.0)
						{
							tmpRight.x = tmpCur.x + 2.0;
							setVertex(i, t + 1, tmpRight);
						}
						else
						{
							break;
						}
					}
					res = false;
				}
			}

		}
	}
	if (res)
	{
		return true;
	}
	else
	{
		return selfCheck();
	}
}

bool isPointInTriangular(const cv::Point2f &pt, const cv::Point2f &V0, const cv::Point2f &V1, const cv::Point2f &V2)
{
	double lambda1 = ((V1.y - V2.y)*(pt.x - V2.x) + (V2.x - V1.x)*(pt.y - V2.y)) / ((V1.y - V2.y)*(V0.x - V2.x) + (V2.x - V1.x)*(V0.y - V2.y));
	double lambda2 = ((V2.y - V0.y)*(pt.x - V2.x) + (V0.x - V2.x)*(pt.y - V2.y)) / ((V2.y - V0.y)*(V1.x - V2.x) + (V0.x - V2.x)*(V1.y - V2.y));
	double lambda3 = 1 - lambda1 - lambda2;
	if (lambda1 >= 0.0 && lambda1 <= 1.0 && lambda2 >= 0.0 && lambda2 <= 1.0 && lambda3 >= 0.0 && lambda3 <= 1.0)
	{
		return true;
	}
	else
	{
		return false;
	}
}

void meshWarp(const cv::Mat src, cv::Mat dst, const Mesh &m1, const Mesh &m2){
	for (int i = 1; i < m1.height; i++)
	{
		for (int j = 1; j < m1.width; j++)
		{
			cv::Point2f p0 = m1.getVertex(i - 1, j - 1);
			cv::Point2f p1 = m1.getVertex(i - 1, j);
			cv::Point2f p2 = m1.getVertex(i, j - 1);
			cv::Point2f p3 = m1.getVertex(i, j);

			cv::Point2f q0 = m2.getVertex(i - 1, j - 1);
			cv::Point2f q1 = m2.getVertex(i - 1, j);
			cv::Point2f q2 = m2.getVertex(i, j - 1);
			cv::Point2f q3 = m2.getVertex(i, j);

			Quad quad1(p0, p1, p2, p3);
			Quad quad2(q0, q1, q2, q3);

			quadWarp(src, dst, quad1, quad2);
		}
	}
}

void quadWarp(const cv::Mat src, cv::Mat dst, const Quad &q1, const Quad &q2){
	int gap = 0;

	int minx = max(0, (int)q2.getMinX());
	int maxx = min(dst.cols - 1, (int)q2.getMaxX());
	int miny = max(0, (int)q2.getMinY());
	int maxy = min(dst.rows - 1, (int)q2.getMaxY());

	/*
	int minx = (int)q2.getMinX();
	int maxx = (int)q2.getMaxX();
	int miny = (int)q2.getMinY();
	int maxy = (int)q2.getMaxY();
	*/

	for (int i = miny; i <= maxy; i++)
	{
		for (int j = minx; j <= maxx; j++)
		{
			if (q2.isPointIn(cv::Point2f(j, i)))
			{
				vector<double> coe;
				bool flag = q2.getBilinearCoordinates(cv::Point2f(j, i), coe);
				if (flag)
				{
					cv::Point2f ptInSrc = q1.getPointByBilinearCoordinates(coe);

					cv::Point2f tmpV00(floor(ptInSrc.x), floor(ptInSrc.y));
					cv::Point2f tmpV01(tmpV00.x + 1, tmpV00.y);
					cv::Point2f tmpV10(tmpV00.x, tmpV00.y + 1);
					cv::Point2f tmpV11(tmpV00.x + 1, tmpV00.y + 1);

					Quad tmpQ(tmpV00, tmpV01, tmpV10, tmpV11);
					vector<double> tmpCoe;
					tmpQ.getBilinearCoordinates(ptInSrc, tmpCoe);

					if (q1.isPointIn(ptInSrc))
					{
						for (int c = 0; c < 3; c++)
						{
							bool f0 = q1.isPointIn(tmpV00);
							bool f1 = q1.isPointIn(tmpV01);
							bool f2 = q1.isPointIn(tmpV10);
							bool f3 = q1.isPointIn(tmpV11);

							dst.at<cv::Vec3b>(i + gap, j + gap)[c] = (tmpCoe[0] * f0*src.at<cv::Vec3b>(int(tmpV00.y), int(tmpV00.x))[c] +
								tmpCoe[1] * f1*src.at<cv::Vec3b>(int(tmpV01.y), int(tmpV01.x))[c] +
								tmpCoe[2] * f2*src.at<cv::Vec3b>(int(tmpV10.y), int(tmpV10.x))[c] +
								tmpCoe[3] * f3*src.at<cv::Vec3b>(int(tmpV11.y), int(tmpV11.x))[c]) /
								(tmpCoe[0] * f0 + tmpCoe[1] * f1 + tmpCoe[2] * f2 + tmpCoe[3] * f3);
						}
					}
				}
			}
		}
	}
}

void meshWarpRemap(cv::Mat &src, cv::Mat &dst, Mesh &m1, Mesh &m2){

	int height = src.size().height;
	int width = src.size().width;
	cv::Mat mapX(height, width, CV_32F); mapX.setTo(0);
	cv::Mat mapY(height, width, CV_32F); mapY.setTo(0);
	vector<cv::Point2f> source(4);
	vector<cv::Point2f> target(4);
	cv::Mat H;

	for (int i = 1; i < m1.height; i++)
	{
		for (int j = 1; j < m1.width; j++)
		{
			Quad s = m1.getQuad(i, j);
			Quad t = m2.getQuad(i, j);

			source[0] = s.V00;
			source[1] = s.V01;
			source[2] = s.V10;
			source[3] = s.V11;

			target[0] = t.V00;
			target[1] = t.V01;
			target[2] = t.V10;
			target[3] = t.V11;

			int minx = max(0, (int)t.getMinX());
			int maxx = min(src.cols - 1, (int)t.getMaxX());
			int miny = max(0, (int)t.getMinY());
			int maxy = min(src.rows - 1, (int)t.getMaxY());

			cv::Mat H_org = cv::findHomography(source, target, 0);
			H = H_org.inv();


			for (int ii = miny; ii <= maxy; ii++)
			{
				for (int jj = minx; jj <= maxx; jj++)
				{
					if (t.isPointIn(cv::Point2f(jj, ii)))
					{
						double x = 1.0*jj;
						double y = 1.0*ii;

						float X = H.at<double>(0, 0) * x + H.at<double>(0, 1) * y + H.at<double>(0, 2);
						float Y = H.at<double>(1, 0) * x + H.at<double>(1, 1) * y + H.at<double>(1, 2);
						float W = H.at<double>(2, 0) * x + H.at<double>(2, 1) * y + H.at<double>(2, 2);

						W = W ? 1.0 / W : 0;

						mapX.at<float>(ii, jj) = X*W;
						mapY.at<float>(ii, jj) = Y*W;
					}
				}
			}


		}
	}
	remap(src, dst, mapX, mapY, cv::INTER_LINEAR, 0, cv::Scalar(0, 0, 0));
}

void Mesh::UpdateMeshByFlow(cv::Mat motionx, cv::Mat motiony)
{
	if (motionx.rows!=height||motionx.cols!=width)
	{
		cerr << "UpdateMeshByFlow ERROR: motion flow do not fit mesh size";
	}
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			cv::Point2f s = this->getVertex(i, j);
			s.x += motionx.at<double>(i, j);
			s.y += motiony.at<double>(i, j);
			
			this->setVertex(i, j, s);
		}
	}
}
