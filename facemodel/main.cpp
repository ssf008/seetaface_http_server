/*
 * http_server_test.cpp
 *
 *  Created on: Oct 26, 2014
 *      Author: liao
 */
#include <sstream>
#include <cstdlib>
#include <unistd.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <dirent.h>
#include <vector>
using namespace std;

#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <opencv2/contrib/contrib.hpp>




#include "simple_log.h"
#include "http_server.h"
#include "threadpool.h"


#include "SeetaFace.h"


//base64
#include <b64/cencode.h>
#include <b64/cdecode.h>
static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

int id_faces_count = 0;
static inline bool is_base64(unsigned char c)
{
    return (isalnum(c) || (c == '+') || (c == '/'));
}
std::string DATA_DIR = "../../data/";
std::string MODEL_DIR = "../../model/";

Detector::Detector(const char* model_name): seeta::FaceDetection(model_name)
{
    this->SetMinFaceSize(40);
    this->SetScoreThresh(2.f);
    this->SetImagePyramidScaleFactor(0.8f);
    this->SetWindowStep(4, 4);
}

SeetaFace::SeetaFace()
{
    this->detector = new Detector((MODEL_DIR+"seeta_fd_frontal_v1.0.bin").c_str());
    this->point_detector = new seeta::FaceAlignment((MODEL_DIR+"seeta_fa_v1.1.bin").c_str());
    this->face_recognizer = new seeta::FaceIdentification((MODEL_DIR+"seeta_fr_v1.0.bin").c_str());
}

float* SeetaFace::NewFeatureBuffer()
{
    return new float[this->face_recognizer->feature_size()];
}

bool SeetaFace::GetFeature(string filename, float* feat,string UserId,string Type,Json::Value &root)
{
    //如果有多张脸，输出第一张脸,把特征放入缓冲区feat，返回true
    //如果没有脸，输出false
    //read pic greyscale

    //filename= "../../cropface/"+UserId+"/"+filename;
    //cout<<"filename = "<<filename<<endl;
    cv::Mat src_img = cv::imread(filename, 0);
    //cout<<"2-------------------"<<endl;
    seeta::ImageData src_img_data(src_img.cols, src_img.rows, src_img.channels());
    src_img_data.data = src_img.data;

    //read pic color
    cv::Mat src_img_color = cv::imread(filename, 1);
    seeta::ImageData src_img_data_color(src_img_color.cols, src_img_color.rows, src_img_color.channels());
    src_img_data_color.data = src_img_color.data;

    std::vector<seeta::FaceInfo> faces = this->detector->Detect(src_img_data);
    cv::Rect face_rect;
    int32_t face_num = static_cast<int32_t>(faces.size());
    cout<<"face_num="<<face_num<<endl;
    if (face_num == 0)
    {
        return false;
    }
    //cout<<"1-------------------"<<endl;

    seeta::FacialLandmark points[5];
    //seeta::FacialLandmark pt5[5];



    //rectangle and crop face
    for (int32_t i = 0; i < face_num; i++)
    {
        face_rect.x = faces[i].bbox.x;
        face_rect.y = faces[i].bbox.y;
        face_rect.width = faces[i].bbox.width;
        face_rect.height = faces[i].bbox.height;


        cv::rectangle(src_img, face_rect, CV_RGB(0, 0, 255), 4, 8, 0);


    }










    //cout<<"2-------------------"<<endl;





    this->point_detector->PointDetectLandmarks(src_img_data, faces[0], points);
    //cout<<"3-------------------"<<endl;
    this->face_recognizer->ExtractFeatureWithCrop(src_img_data_color, points, feat);
    //cout<<"4-------------------"<<endl;
#if 0
    for (int i = 0; i<5; i++)
    {
        cv::circle(src_img, cv::Point(points[i].x, points[i].y), 2,
                   CV_RGB(0, 255, 0));
        cv::circle(src_img_color, cv::Point(points[i].x, points[i].y), 2,
                   CV_RGB(0, 255, 0));
    }
    cv::imwrite("result.jpg", src_img);
    cv::namedWindow("test",CV_WINDOW_AUTOSIZE);
    cv::Mat img = cv::imread("result.jpg");
    cv::imshow("test",img);
    cv::waitKey(0);
#endif

    return true;
};


bool SeetaFace::GetFeature(cv::Mat mat_img, float* feat,string UserId,string Type,Json::Value &root)
{
    cv::Mat img_gray;
    if (mat_img.channels() != 1)
        cv::cvtColor(mat_img, img_gray, cv::COLOR_BGR2GRAY);
    else
        img_gray = mat_img;

    seeta::ImageData img_data;
    img_data.data = img_gray.data;
    img_data.width = img_gray.cols;
    img_data.height = img_gray.rows;
    img_data.num_channels = 1;

    cout<<"1-------------------"<<endl;
    std::vector<seeta::FaceInfo> faces = detector->Detect(img_data);

    cv::Rect face_rect;
    int32_t num_face = static_cast<int32_t>(faces.size());

    if (num_face == 0)
    {
        cout<<"no fund face="<<endl;
        root["Tatus"] = "error";
        return false;
    }


    seeta::FacialLandmark points[5];

    IplImage img_colortmp = mat_img;
    IplImage *img_clone = cvCloneImage(&img_colortmp);

    for (int32_t i = 0; i < num_face; i++)
    {
        face_rect.x = faces[i].bbox.x;
        face_rect.y = faces[i].bbox.y;
        face_rect.width = faces[i].bbox.width;
        face_rect.height = faces[i].bbox.height;
        cout<<face_rect.x<<","<<face_rect.y<<","<<face_rect.width<<","<<face_rect.height<<endl;
        cv::rectangle(mat_img, face_rect, CV_RGB(0, 0, 255), 4, 8, 0);
        cvSetImageROI(img_clone,cvRect(face_rect.x,face_rect.y,face_rect.width,face_rect.height));
    }
#if 0
    cv::imshow("img",mat_img);
    cv::waitKey(0);
#endif // 0
    if(Type=="0")
    {
        cout<<"2-------------------"<<endl;
        cv::Mat myimg(img_clone,0);

        time_t curtime = time(NULL);
        //time(&curtime);
        stringstream ss;
        ss<<curtime;
        string s1 = ss.str();

        std::cout << UserId+"_"+s1  << std::endl;

//mkdir
        string dir="../../cropface/"+UserId;
        if (access(dir.c_str(), 0) == -1)
        {
            id_faces_count = 0;
            cout<<dir<<" is not existing"<<endl;
            cout<<"now make it"<<endl;
            int flag=mkdir(dir.c_str(), 0777);

            if (flag == 0)
            {
                cout<<"make successfully"<<endl;
            }
            else
            {
                cout<<"make errorly"<<endl;
            }
        }
        if(id_faces_count<5)
        {
            cv::imwrite("../../cropface/"+UserId+"/"+UserId+"_"+s1+".jpg", myimg);
            id_faces_count++;
            root["Tatus"] = "sucess";
        }
        else
        {
            root["Tatus"] = 1005;
        }




        cout<<"current id_faces_count is "<<id_faces_count<<endl;

        root["OptImagsName"] = UserId+"_"+s1+".jpg";
        root["Message"] = id_faces_count;
        root["UserId"] = UserId;

    }
    else if(Type=="1")
    {

        //cout<<"1=-------------------"<<endl;

        this->point_detector->PointDetectLandmarks(img_data, faces[0], points);
        //cout<<"2=-------------------"<<endl;
        this->face_recognizer->ExtractFeatureWithCrop(img_data, points, feat);//base64 error
        //cout<<"3=-------------------"<<endl;
    }


    return true;

}

int SeetaFace::GetFeatureDims()
{
    return this->face_recognizer->feature_size();
}

float SeetaFace::FeatureCompare(float* feat1, float* feat2)
{
    return this->face_recognizer->CalcSimilarity(feat1, feat2);
}


//base64
std::string base64_decode(std::string const& encoded_string)
{
    int in_len = encoded_string.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::string ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_]))
    {
        char_array_4[i++] = encoded_string[in_];
        in_++;
        if (i == 4)
        {
            for (i = 0; i < 4; i++)
                char_array_4[i] = base64_chars.find(char_array_4[i]);

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; (i < 3); i++)
                ret += char_array_3[i];
            i = 0;
        }
    }

    if (i)
    {
        for (j = i; j < 4; j++)
            char_array_4[j] = 0;

        for (j = 0; j < 4; j++)
            char_array_4[j] = base64_chars.find(char_array_4[j]);

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; (j < i - 1); j++) ret += char_array_3[j];
    }

    return ret;
}

/***** Global Variables *****/



/* Show all files under dir_name , do not show directories ! */


void HexToAscii(const char * hex, int length, char * ascii)
{
    for (int i = 0; i < length; i += 2)
    {
        if (hex[i] >= '0' && hex[i] <= '9')
            ascii[i / 2] = (hex[i] - '0') << 4;
        else if (hex[i] >= 'a' && hex[i] <= 'z')
            ascii[i / 2] = (hex[i] - 'a' + 10) << 4;
        else if (hex[i] >= 'A' && hex[i] <= 'Z')
            ascii[i / 2] = (hex[i] - 'A' + 10) << 4;

        if (hex[i + 1] >= '0' && hex[i + 1] <= '9')
            ascii[i / 2] += hex[i + 1] - '0';
        else if (hex[i + 1] >= 'a' && hex[i + 1] <= 'z')
            ascii[i / 2] += hex[i + 1] - 'a' + 10;
        else if (hex[i + 1] >= 'A' && hex[i + 1] <= 'Z')
            ascii[i / 2] += hex[i + 1] - 'A' + 10;
    }
}

void AsciiToHex(const char * ascii, int length, char * hex)
{
    static const char plate[] = "0123456789abcdef";

    for (int i = 0; i < length; i++)
    {
        hex[i * 2 + 1] = plate[ascii[i] & 0x0F];
        hex[i * 2] = plate[(ascii[i] >> 4) & 0x0F];
    }
}


void addface(Request &request, Json::Value &root)
{
    std::string UserId = request.get_param("UserId");
    std::string Type = request.get_param("Type");
    //int iscrop = atoi(Type.c_str());
    std::string ImgsBase64 = request.get_param("ImgsBase64");//Type = 0 need
    std::string SrcUrls = request.get_param("SrcUrls");//Type = 1 need
    std::string OptUrls = request.get_param("OptUrls");
    std::string Message = request.get_param("Message");
    cout<<"request.get_method="<<request.get_method()<<endl;

    cout<<"Type="<<Type<<endl;
    cout<<"UserId="<<UserId<<endl;

    LOG_DEBUG("login user which UserId:%s, Type:%s, ImgsBase64:%s, SrcUrls=%s,OptUrls=%s,Message=%s", UserId.c_str(), Type.c_str(),ImgsBase64.c_str(),SrcUrls.c_str(),OptUrls.c_str(),Message.c_str());

    //ImgsBase64 = "61626364656161616161616161616161616161616161616161736461736461737373737373737373737373737373737373737373737373";

    char base64toascii[2000000];
    //char urlstoascii[100];
    SeetaFace sf;
    int myflag = 1;
    float maxsimi;
    float *feat1=sf.NewFeatureBuffer(),*feat2=sf.NewFeatureBuffer();

#if 0
    if(ImgsBase64=="")
    {
        ImgsBase64="2f396a2f34";
    }
#endif
    HexToAscii(ImgsBase64.c_str(), ImgsBase64.length(), base64toascii);
    //HexToAscii(SrcUrls.c_str(), SrcUrls.length(), urlstoascii);
    std::string decoded_string = base64_decode(base64toascii);
    std::vector<uchar> data(decoded_string.begin(), decoded_string.end());
    cv::Mat decode_img = cv::imdecode(data, cv::IMREAD_UNCHANGED);



    if(Type=="0")
    {
        cout<<"enter=====qu yang==="<<endl;


#if 0
        if (!decode_img.empty())
        {
            cv::imshow("decode_img",decode_img);
            cv::waitKey(0);
        }
#endif

//crop face

        sf.GetFeature(decode_img, feat1,UserId,Type,root);








    }

//similarity---------------------------------------

    if(Type=="1")
    {
        cout<<"enter=====simi==="<<endl;

        cv::imwrite("newface.jpg", decode_img);
        sf.GetFeature("newface.jpg", feat1,UserId,Type,root);




        string mypath =  "../../cropface/"+UserId;

        struct dirent *direntp;
        char imgsurl[100];
        //string imgurl;
        // string & myimgs = imgurl;
        DIR *dirp = opendir(mypath.c_str());

        if (dirp != NULL)
        {
            while ((direntp = readdir(dirp)) != NULL)
            {
                if( strcmp( direntp->d_name , "." ) == 0 ||
                        strcmp( direntp->d_name , "..") == 0    )
                    continue;


                strcpy(imgsurl,direntp->d_name);
                //imgurl = imgsurl;
                //printf( imgurl);
#if 0
                if (i>4)
                {
                    cout << "error,imgs > 5"<<endl;
                }
#endif // 0
                cout<<"imgurl  :"<<"../../cropface/"+UserId+"/"+imgsurl<<endl;

                sf.GetFeature("../../cropface/"+UserId+"/"+imgsurl, feat2,UserId,Type,root);
                cout<<"2========="<<endl;
                float simi = sf.FeatureCompare(feat1, feat2);
                //init maxsimi
                if (myflag)
                {
                    maxsimi = simi;
                    myflag = 0;
                }

                if(simi>maxsimi)
                {
                    maxsimi = simi;
                }

                //i++;


            }
        }

        closedir(dirp);


        std::cout<<"Similarity(0-1):"<<maxsimi<<std::endl;
        root["UserId"] = UserId;
        root["Similarity"] = maxsimi;
        root["Message"] = "sucess";
    }

}


int main(int argc, char **argv)
{
    int ret = log_init("../../conf/", "simple_log.conf");
    if (ret != 0)
    {
        printf("log init error!");
        return 0;
    }
#if 1
    if (argc < 3)
    {
        cout << "Usage: " << argv[0]
             << "[IP]"<<argv[1]
             << "[Port]"<<argv[2]
             << endl;
        return -1;
    }
#endif






    //pthread_key_create(&g_tp_key,NULL);
    //
    //ThreadPool tp;
    //tp.set_thread_start_cb(a_test_fn);
    //tp.set_pool_size(4);

    HttpServer http_server;
    //http_server.set_thread_pool(&tp);


    http_server.add_mapping("/addface", addface, GET_METHOD | POST_METHOD);

    //http_server.add_bind_ip("192.168.0.230");
    http_server.add_bind_ip(argv[1]);
    http_server.set_port(atoi(argv[2]));
    http_server.set_backlog(100000);
    http_server.set_max_events(100000);
    //http_server.add_bind_ip("192.168.238.158");

    http_server.start_async();
    //sleep(1);
    //http_server.stop();
    http_server.join();



    return 0;
}
