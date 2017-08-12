////
////  main.cpp
////  HandDetectUsingCircle
////
////  Created by 하정수 on 12/08/2017.
////  Copyright © 2017 하정수. All rights reserved.
////
//
//#include <iostream>
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/opencv.hpp"
//
//#include <math.h>
//#include <stdlib.h>
//#include <stdio.h>
//
//using namespace std;
//using namespace cv;
//
////방법1.
////반복문으로 각 화소 모두 비교하는 방법
//Mat getHandMask1(const Mat& image, int minCr=128, int maxCr=170, int minCb=73, int maxCb=158){
//    //컬러 공간 변환 BGR->YCrCb
//    Mat YCrCb;
//    cvtColor(image, YCrCb, CV_BGR2YCrCb);
//
//    //각 채널별로 분리
//    vector<Mat> planes;
//    split(YCrCb, planes);
//
//    //각 채널별로 화소마다 비교
//    Mat mask(image.size(), CV_8U, Scalar(0));   //결과 마스크를 저장할 영상
//    int nr=image.rows;    //전체 행의 수
//    int nc=image.cols;
//
//    for(int i=0; i<nr; i++){
//        uchar* CrPlane=planes[1].ptr<uchar>(i);   //Cr채널의 i번째 행 주소
//        uchar* CbPlane=planes[2].ptr<uchar>(i);   //Cb채널의 i번째 행 주소
//        for(int j=0; j<nc; j++){
//            if( (minCr < CrPlane[j]) && (CrPlane[j] <maxCr) && (minCb < CbPlane[j]) && (CbPlane[j] < maxCb) )
//                mask.at<uchar>(i, j)=255;
//        }
//    }
//
//    return mask;
//}
//
////방법2.
////비교연산자, 논리 연산자를 이용한 방법
//Mat getHandMask2(const Mat& image, int minCr=128, int maxCr=170, int minCb=73, int maxCb=158){
//    //컬러 공간 변환 BGR->YCrCb
//    Mat YCrCb;
//    cvtColor(image, YCrCb, CV_BGR2YCrCb);
//
//    //각 채널별로 분리
//    vector<Mat> planes;
//    split(YCrCb, planes);
//
//    Mat mask=(minCr < planes[1]) & (planes[1] < maxCr) & (minCb < planes[2]) & (planes[2] < maxCb);
//
//    return mask;
//}
//
////손바닥의 중심점과 반지름 반환
////입력은 8bit 단일 채널(CV_8U), 반지름을 저장할 double형 변수
//Point getHandCenter(const Mat& mask, double& radius){
//    //거리 변환 행렬을 저장할 변수
//    Mat dst;
//    distanceTransform(mask, dst, CV_DIST_L2, 5);  //결과는 CV_32SC1 타입
//
//    //거리 변환 행렬에서 값(거리)이 가장 큰 픽셀의 좌표와, 값을 얻어온다.
//    int maxIdx[2];    //좌표 값을 얻어올 배열(행, 열 순으로 저장됨)
//    minMaxIdx(dst, NULL, &radius, NULL, maxIdx, mask);   //최소값은 사용 X
//
//    return Point(maxIdx[1], maxIdx[0]);
//}
//
//int getFingerCount(const Mat& mask, Point center, double radius, double scale=2.0){
//    //손가락 개수를 세기 위한 원 그리기
//    Mat cImg(mask.size(), CV_8U, Scalar(0));
//    circle(cImg, center, radius*scale, Scalar(255));
//
//    //원의 외곽선을 저장할 벡터
//    vector<vector<Point>> contours;
//    findContours(cImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
//
//    if(contours.size()==0)   //외곽선이 없을 때 == 손 검출 X
//        return -1;
//
//    //외곽선을 따라 돌며 mask의 값이 0에서 1로 바뀌는 지점 확인
//    int fingerCount=0;
//    for(int i=1; i<contours[0].size(); i++){
//        Point p1=contours[0][i-1];
//        Point p2=contours[0][i];
//        if(mask.at<uchar>(p1.y, p1.x)==0 && mask.at<uchar>(p2.y, p2.x)>1)
//            fingerCount++;
//    }
//
//    //손목과 만나는 개수 1개 제외
//    return fingerCount-1;
//}
//
//int main(){
//
//    Mat image = imread("/Users/jeongsooha/MyDesktop/testpictures/5.jpeg");
//  
//               Mat mask=getHandMask1(image);
//        erode(mask, mask, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2);
//        double radius;
//        Point center=getHandCenter(mask, radius);
//
//        cout<<"손바닥 중심점 좌표:"<<center<<", 반지름:"<<radius<<"손가락 개수"<<getFingerCount(mask, center, radius)<<endl;
//
//        //손바닥 중심점 그리기
//        circle(image, center, 2, Scalar(0, 255, 0), -1);
//
//        //손바닥 영역 그리기
//        circle(image, center, (int)(radius+2.0), Scalar(255, 0, 0), 2);
//
//        cv::resize( image, image, cv::Size( 450, 600), 0, 0, CV_INTER_NN );
//        cv::resize( mask, mask, cv::Size( 450, 600), 0, 0, CV_INTER_NN );
//
//
//        namedWindow( "Original Image With Hand Center", CV_WINDOW_AUTOSIZE );
//        namedWindow( "mask", CV_WINDOW_AUTOSIZE );
//
////        cvResizeWindow("Original Image With Hand Center", 350, 350);
////        cvResizeWindow("mask", 350, 350);
//
//        imshow("Original Image With Hand Center", image);
//        imshow("mask", mask);
//
//
//
//        waitKey(0);
//
//
//  //  }
//
//
//}
//
