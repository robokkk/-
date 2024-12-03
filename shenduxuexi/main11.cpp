#include <iostream>
#include <iomanip>
#include "inference.h"
#include <filesystem>
#include <fstream>
#include <random>

void Detector(YOLO_V8*& p) {

            cv::VideoCapture cap;
            cap.open(1);
            cv::Mat img;
            if(!cap.isOpened())
                std::cout<<"aaa";
            while (cap.read(img)) {
            //capture >> img;
            //cv::Mat img = cv::imread("D:/Python/2.jpg");
            std::vector<DL_RESULT> res;
            p->RunSession(img, res);

            for (auto& re : res)
            {
                cv::RNG rng(cv::getTickCount());
                cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
                cv::rectangle(img, re.box, color, 3);

                float confidence = floor(100 * re.confidence) / 100;
                std::cout << std::fixed << std::setprecision(2);
                std::string label = p->classes[re.classId] + " " +    std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

                cv::rectangle(img,cv::Point(re.box.x, re.box.y - 25),cv::Point(re.box.x + label.length() * 15, re.box.y),color,cv::FILLED);
                cv::putText(img,label,cv::Point(re.box.x, re.box.y - 5),cv::FONT_HERSHEY_SIMPLEX,0.75,cv::Scalar(0, 0, 0),2);


            }



            cv::imshow("Result of Detection", img);
            char cc = (char)cv::waitKey(10);
            if (cc == 27) break;
            }
            cap.release();
            cv::destroyAllWindows();
}



void DetectTest()
{
    YOLO_V8* yoloDetector = new YOLO_V8;

    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.1;
    params.iouThreshold = 0.5;
    params.modelPath = "D:/pytorch project/YOLOv8-ONNXRuntime-CPP/best(1).onnx";
    params.imgSize = { 640, 640 };
#ifdef USE_CUDA
    params.cudaEnable = true;

    // GPU FP32 inference
    params.modelType = YOLO_DETECT_V8;
    // GPU FP16 inference
    //Note: change fp16 onnx model
    //params.modelType = YOLO_DETECT_V8_HALF;

#else
    // CPU inference
    params.modelType = YOLO_DETECT_V8;
    params.cudaEnable = false;

#endif
    yoloDetector->CreateSession(params);
    Detector(yoloDetector);
}



int main()
{   
    
    DetectTest();
}
