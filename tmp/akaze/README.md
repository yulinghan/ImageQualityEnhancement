akaze-opencv
============

# AKAZE is officially supported by OpenCV since 3.0.0 alpha. Please use it instead of this repo.

wrap AKAZE features implementatino to cv::Feature2D API without rebuilding OpenCV

## Usage
You can use AKAZE feature detector and descriptor extractor as OpenCV common ones.

```
Mat img = imread(...);
std::vector<KeyPoint> keypoints;
Mat descriptors;
Ptr<FeatureDetector> detector = FeatureDetector::create("AKAZE");
detector->detect(img, keypoints);
Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("AKAZE");
extractor->compute(img, keypoints, descriptors);
```

See [main.cpp](main.cpp) for more detail.

## License
Files under `akaze` folder are developed by Pablo F. Alcantarilla and Jesus Nuevo. Please see [akaze/LICENSE](akaze/LICENSE) as their license.

Other files are developed by Takahiro "Poly" Horikawa. Their license are public domain.
