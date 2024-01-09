#include "MeshFlow.h"

MeshFlow::MeshFlow(int width, int height){
	m_height = height;
	m_width = width;
	m_divide_x = 16;
	m_divide_y = 16;
	m_quadWidth = 1.0*m_width / pow(2.0, 4);
	m_quadHeight = 1.0*m_height / pow(2.0, 4);

	m_mesh = new Mesh(m_height, m_width, 1.0*m_quadWidth, 1.0*m_quadHeight);
	m_warpedmesh = new Mesh(m_height, m_width, 1.0*m_quadWidth, 1.0*m_quadHeight);
	m_meshheight = m_mesh->height;
	m_meshwidth = m_mesh->width;
	m_vertexMotion.resize(m_meshheight*m_meshwidth);
	m_vertexMotionMatX.create(m_meshheight, m_meshwidth, CV_64FC1);
	m_vertexMotionMatY.create(m_meshheight, m_meshwidth, CV_64FC1);
}

void MeshFlow::ReInitialize(){
	for (int i = 0; i < m_meshheight*m_meshwidth; i++) m_vertexMotion[i].x = m_vertexMotion[i].y = 0;
	n.features.clear();
	n.motions.clear();
	n.MeshMotions.clear();
}

void MeshFlow::SetFeature(vector<cv::Point2f> &spt, vector<cv::Point2f> &tpt){

	//global outlier rejection
	vector<int> mask;
	m_globalHomography = findHomography(spt, tpt, mask, cv::RANSAC);

	vector<cv::Point2f> spt_prune,tpt_prune;
	for (int i = 0; i < mask.size(); i++){
		if (mask[i]){
			spt_prune.push_back(spt[i]);
			tpt_prune.push_back(tpt[i]);
		}
	}
	
	n.features = tpt_prune;
	n.MeshMotions.resize(m_divide_x*m_divide_y);
	
	n.location_x.resize(tpt_prune.size());
	n.location_y.resize(tpt_prune.size());
	for (int i = 0; i < tpt_prune.size(); i++){
		if (tpt_prune[i].x<0 || tpt_prune[i].x>m_width - 1 || tpt_prune[i].y<0 || tpt_prune[i].y>m_height - 1) {
			continue;
		} else {
			n.location_x[i] = tpt_prune[i].x / m_quadWidth;
			n.location_y[i] = tpt_prune[i].y / m_quadHeight;
			n.MeshMotions[n.location_x[i] + n.location_y[i] * m_divide_x].push_back(tpt_prune[i] - spt_prune[i]);
		}
	}
}

void MeshFlow::Execute(int radius){
	DistributeMotion2MeshVertexes_MedianFilter();  //the first median filter
	SpatialMedianFilter(radius); //the second median filter
	WarpMeshbyMotion();
}

void MeshFlow::DistributeMotion2MeshVertexes_MedianFilter(){

    //使用全局配准结果，对每个网格点进行图像配准，并保存配准偏移参数
	for (int i = 0; i < m_meshheight; i++){
		for (int j = 0; j < m_meshwidth; j++){
			cv::Point2f pt = m_mesh->getVertex(i, j);
			cv::Point2f pttrans = Trans(m_globalHomography, pt);
			m_vertexMotion[i*m_meshwidth + j].x = pttrans.x - pt.x;
			m_vertexMotion[i*m_meshwidth + j].y = pttrans.y - pt.y;
		}
	}

	vector<vector<float>> motionx, motiony;
	motionx.resize(m_meshheight*m_meshwidth);
	motiony.resize(m_meshheight*m_meshwidth);

	//n.MeshMotions保存的是：落在对应网格里面中的所有待配准图像特征点，和它配对特征点的像素偏移。
    //这里是扩大每个网格特征点收集范围：以当前特征点为中心，将周围5x5范围网格内的特征点全部整合到当前网格队列motionx/motiony中。
	for (int i = 0; i < m_meshheight; i++){
		for (int j = 0; j < m_meshwidth; j++){
			for (int mm = i - 2; mm <= i + 1; mm++) {
				for (int nn = j - 2; nn <= j + 1; nn++) {
					if (mm>=0&&nn>=0&&mm<m_meshheight-1&&nn<m_meshwidth-1) {
						for (int ii = 0; ii < n.MeshMotions[mm*m_divide_x + nn].size(); ii++) {
							motionx[i*m_meshwidth + j].push_back(n.MeshMotions[mm*m_divide_x + nn][ii].x);
							motiony[i*m_meshwidth + j].push_back(n.MeshMotions[mm*m_divide_x + nn][ii].y);
						}
					}
				}
			}
		}
	}

    //如果当前网格队列特征点数量大于4个。
    //对motionx和motiony中的特征点偏移数据进行从小到大排序。
    //取中间值保存到m_vertexMotion中。
    //特别注意：m_vertexMotion在之前已经用全局配准数据赋值初始化过了。
    //这里的实际意思是：如果当前网格内有大于4个特征点，那个就用特征点数据替换全局配准结果，如果没有，那个当前网格就用全局配准数据。
	for (int i = 0; i < m_meshheight; i++){
		for (int j = 0; j<m_meshwidth; j++){
			if (motionx[i*m_meshwidth + j].size()>4){

				myQuickSort(motionx[i*m_meshwidth + j], 0, motionx[i*m_meshwidth + j].size() - 1);
				myQuickSort(motiony[i*m_meshwidth + j], 0, motiony[i*m_meshwidth + j].size() - 1);

				m_vertexMotion[i*m_meshwidth + j].x = motionx[i*m_meshwidth + j][motionx[i*m_meshwidth + j].size() / 2];
				m_vertexMotion[i*m_meshwidth + j].y = motiony[i*m_meshwidth + j][motiony[i*m_meshwidth + j].size() / 2];
			}
		}
	}
}

void MeshFlow::SpatialMedianFilter(int radius){
	VecPt2f tempVertexMotion(m_meshheight*m_meshwidth);
	for (int i = 0; i < m_meshheight; i++){
		for (int j = 0; j < m_meshwidth; j++){
			tempVertexMotion[i*m_meshwidth + j] = m_vertexMotion[i*m_meshwidth + j];
		}
	}


    //根据预设滤波半径radius。
    //以当前网格点为中心，聚合半径范围内的特征偏移数据为列表。
    //对列表根据特征偏移数据，从小到大排序，取中间值作为当前网格点特征偏移。
	for (int i = 0; i < m_meshheight; i++){
		for (int j = 0; j < m_meshwidth; j++){

			vector<float> motionx;
			vector<float> motiony;

			for (int k = -radius; k <= radius; k++){
				for (int l = -radius; l <= radius; l++){
					if (k + i >= 0 && (k + i) < m_meshheight && (l + j) >= 0 && (l + j) < m_meshwidth){
						motionx.push_back(tempVertexMotion[(k + i)*m_meshwidth + l + j].x);
						motiony.push_back(tempVertexMotion[(k + i)*m_meshwidth + l + j].y);
					}
				}
			}
			myQuickSort(motionx, 0, motionx.size() - 1);
			myQuickSort(motiony, 0, motiony.size() - 1);
			m_vertexMotion[i*m_meshwidth + j].x = motionx[motionx.size() / 2];
			m_vertexMotion[i*m_meshwidth + j].y = motiony[motiony.size() / 2];
		}
	}
}

void MeshFlow::WarpMeshbyMotion(){

    //将每个网格点偏移数据和原始坐标点相加，得到每个网格点真实位移数据
	for (int i = 0; i < m_meshheight; i++){
		for (int j = 0; j < m_meshwidth; j++){
			cv::Point2f s = m_mesh->getVertex(i, j);
			s += m_vertexMotion[i*m_meshwidth + j];
			m_warpedmesh->setVertex(i, j, s);
		}
	}
}

void MeshFlow::GetMotions(cv::Mat &mapX, cv::Mat &mapY){

	vector<cv::Point2f> source(4);
	vector<cv::Point2f> target(4);
	cv::Mat H;

	for (int i = 1; i < m_mesh->height; i++)
	{
		for (int j = 1; j < m_mesh->width; j++)
		{
			Quad s = m_mesh->getQuad(i, j);
			Quad t = m_warpedmesh->getQuad(i, j);

			source[0] = s.V00;
			source[1] = s.V01;
			source[2] = s.V10;
			source[3] = s.V11;

			target[0] = t.V00;
			target[1] = t.V01;
			target[2] = t.V10;
			target[3] = t.V11;

			H = cv::findHomography(source, target, 0);

			for (int ii = source[0].y; ii < source[3].y; ii++){
				for (int jj = source[0].x; jj < source[3].x; jj++){
					double x = 1.0*jj;
					double y = 1.0*ii;

					double X = H.at<double>(0, 0) * x + H.at<double>(0, 1) * y + H.at<double>(0, 2);
					double Y = H.at<double>(1, 0) * x + H.at<double>(1, 1) * y + H.at<double>(1, 2);
					double W = H.at<double>(2, 0) * x + H.at<double>(2, 1) * y + H.at<double>(2, 2);

					W = W ? 1.0 / W : 0;
					mapX.at<double>(ii, jj) = X*W - jj;
					mapY.at<double>(ii, jj) = Y*W - ii;
				}
			}
		}
	}
}

void MeshFlow::GetWarpedSource(cv::Mat &dst){
	meshWarpRemap(m_source, dst, *m_mesh, *m_warpedmesh);
}

void MeshFlow::GetWarpedSource(cv::Mat &source, cv::Mat &dst){
	meshWarpRemap(source, dst, *m_mesh, *m_warpedmesh);
}

cv::Point2f MeshFlow::Trans(cv::Mat H, cv::Point2f &pt){
	cv::Point2f result;

	float a = H.at<double>(0, 0) * pt.x + H.at<double>(0, 1) * pt.y + H.at<double>(0, 2);
	float b = H.at<double>(1, 0) * pt.x + H.at<double>(1, 1) * pt.y + H.at<double>(1, 2);
	float c = H.at<double>(2, 0) * pt.x + H.at<double>(2, 1) * pt.y + H.at<double>(2, 2);

	result.x = a / c;
	result.y = b / c;

	return result;
}

cv::Mat MeshFlow::GetVertexMotionX(){
	for (int i = 0; i < m_meshheight; i++){
		for (int j = 0; j < m_meshwidth; j++){
			m_vertexMotionMatX.at<double>(i, j) = m_vertexMotion[i*m_meshwidth + j].x;
		}
	}
	return m_vertexMotionMatX;
}

cv::Mat MeshFlow::GetVertexMotionY(){
	for (int i = 0; i < m_meshheight; i++){
		for (int j = 0; j < m_meshwidth; j++){
			m_vertexMotionMatY.at<double>(i, j) = m_vertexMotion[i*m_meshwidth + j].y;
		}
	}
	return m_vertexMotionMatY;
}

void MeshFlow::SetVertexMotion(const cv::Mat motionx, const cv::Mat motiony)
{
	for (int i = 0; i < m_meshheight; i++){
		for (int j = 0; j < m_meshwidth; j++){
			m_vertexMotion[i*m_meshwidth + j].y = motiony.at<double>(i, j);
			m_vertexMotion[i*m_meshwidth + j].x = motionx.at<double>(i, j);
		}
	}
	WarpMeshbyMotion();
}

MeshFlow::~MeshFlow(){
	delete m_mesh;
	delete m_warpedmesh;
}


void myQuickSort(vector<float> &arr, int left, int right){
	int i = left, j = right;
	double tmp;
	double pivot = arr[(left + right) / 2];

	while (i <= j){
		while (arr[i]<pivot)
			i++;
		while (arr[j]>pivot)
			j--;
		if (i <= j){
			tmp = arr[i];
			arr[i] = arr[j];
			arr[j] = tmp;
			i++;
			j--;
		}
	}
	if (left < j)myQuickSort(arr, left, j);
	if (i < right)myQuickSort(arr, i, right);
}


