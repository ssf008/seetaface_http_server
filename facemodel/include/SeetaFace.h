#include <string>
#include<iostream>
using std::string;
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "face_detection.h"
#include "face_alignment.h"
#include "face_identification.h"

class Detector : public seeta::FaceDetection{
public:
	Detector(const char * model_name);
};

class SeetaFace{
public:
	SeetaFace();
	Detector* detector;
	seeta::FaceAlignment* point_detector;
	seeta::FaceIdentification* face_recognizer;
	bool GetFeature(string filename, float* feat,string useraccount,string Type,Json::Value &root);
	bool GetFeature(cv::Mat mat_img, float* feat,string UserId,string Type,Json::Value &root);
	float* NewFeatureBuffer();
	float FeatureCompare(float* feat1, float* feat2);
	int GetFeatureDims();
};
