//
//  main.cpp
//  HandDetectUsingCircle
//
//  Created by 하정수 on 12/08/2017.
//  Copyright © 2017 하정수. All rights reserved.
//

#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

using namespace std;
using namespace cv;

// Method to get Skin Tone
Mat getHandMask1(const Mat& image){
    //컬러 공간 변환 BGR->YCrCb
    int Y_MIN = 0;
    int Y_MAX = 255;
    int Cr_MIN = 133;
    int Cr_MAX = 173;
    int Cb_MIN = 77;
    int Cb_MAX = 127;
    Mat YCrCb;
    
    // Chaning the Image Format
    cvtColor(image, YCrCb, COLOR_BGR2YCrCb);
    
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
    
    //    return mask;
    
    /*
    imshow("YCrCb Color Space", YCrCb);
    */
    
    // Make the test if Color of Skin meets defined color
    inRange(YCrCb,Scalar(Y_MIN,Cr_MIN,Cb_MIN),Scalar(Y_MAX,Cr_MAX,Cb_MAX),YCrCb);
    
    
    
    return YCrCb;
    
}


double calcDistance(const CvPoint& p1,const CvPoint& p2){
    double distance;
    distance = sqrt((p1.x - p1.x,2) + pow(p1.y - p2.y,2));
    return distance;
}

double calcDistance(const CvPoint& p1,const Point& p2){
    double distance;
    distance = sqrt((p1.x - p1.x,2) + pow(p1.y - p2.y,2));
    return distance;
}

////No use Function
//비교연산자, 논리 연산자를 이용한 방법
Mat getHandMask2(const Mat& image, int minCr=128, int maxCr=170, int minCb=73, int maxCb=158){
    //컬러 공간 변환 BGR->YCrCb
    Mat YCrCb;
    cvtColor(image, YCrCb, CV_BGR2YCrCb);
    
    //각 채널별로 분리
    vector<Mat> planes;
    split(YCrCb, planes);
    
    Mat mask=(minCr < planes[1]) & (planes[1] < maxCr) & (minCb < planes[2]) & (planes[2] < maxCb);
    
    return mask;
}

//손바닥의 중심점과 반지름 반환
//입력은 8bit 단일 채널(CV_8U), 반지름을 저장할 double형 변수
Point getHandCenter(const Mat& mask, double& radius){
    //거리 변환 행렬을 저장할 변수
    Mat dst;
    distanceTransform(mask, dst, CV_DIST_L2, 5);  //결과는 CV_32SC1 타입
    
    //거리 변환 행렬에서 값(거리)이 가장 큰 픽셀의 좌표와, 값을 얻어온다.
    int maxIdx[2];    //좌표 값을 얻어올 배열(행, 열 순으로 저장됨)
    minMaxIdx(dst, NULL, &radius, NULL, maxIdx, mask);   //최소값은 사용 X
    
    return Point(maxIdx[1], maxIdx[0]);
}

int getFingerCount(const Mat& mask, Point center, double radius, double scale=2.0){
    //손가락 개수를 세기 위한 원 그리기
    Mat cImg(mask.size(), CV_8U, Scalar(0));
    circle(cImg, center, radius*scale, Scalar(255));
    
    //원의 외곽선을 저장할 벡터
    vector<vector<Point>> contours;
    findContours(cImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    
    if(contours.size()==0)   //외곽선이 없을 때 == 손 검출 X
        return -1;
    
    //외곽선을 따라 돌며 mask의 값이 0에서 1로 바뀌는 지점 확인
    int fingerCount=0;
    for(int i=1; i<contours[0].size(); i++){
        Point p1=contours[0][i-1];
        Point p2=contours[0][i];
        if(mask.at<uchar>(p1.y, p1.x)==0 && mask.at<uchar>(p2.y, p2.x)>1)
            fingerCount++;
    }
    
    //손목과 만나는 개수 1개 제외
    return fingerCount-1;
}



void  detect(IplImage* imgTonedImage,IplImage* imgRealFeed, const Point& center) {
    CvMemStorage* storage = cvCreateMemStorage();
    CvSeq* first_contour = NULL;
    CvSeq* maxitem=NULL;
    double area=0,areamax=0;
    int maxn=0;
    int Nc = cvFindContours(imgTonedImage,storage,&first_contour,sizeof(CvContour),CV_RETR_LIST);
    int n=0;
    if(Nc>0){
        for( CvSeq* c=first_contour; c!=NULL; c=c->h_next ){
            area=cvContourArea(c,CV_WHOLE_SEQ );
            if(area>areamax){
                areamax=area;
                maxitem=c;
                maxn=n;
            }
            n++;
        }
        CvMemStorage* storage3 = cvCreateMemStorage(0);
        if(areamax>5000){
            maxitem = cvApproxPoly( maxitem, sizeof(CvContour), storage3, CV_POLY_APPROX_DP, 10, 1 );
            CvPoint pt0;
            CvMemStorage* storage1 = cvCreateMemStorage(0);
            CvMemStorage* storage2 = cvCreateMemStorage(0);
            CvSeq* ptseq = cvCreateSeq( CV_SEQ_KIND_GENERIC|CV_32SC2, sizeof(CvContour), sizeof(CvPoint), storage1 );
            CvSeq* hull;
            CvSeq* defects;
            //
            for(int i = 0; i < maxitem->total; i++ ){
                CvPoint* p = CV_GET_SEQ_ELEM( CvPoint, maxitem, i );
                pt0.x = p->x;
                pt0.y = p->y;
                cvSeqPush( ptseq, &pt0 );
                //cvCircle( imgRealFeed, *p, 5, CV_RGB(0,0,50*i), 2, 8,0);
            }
            hull = cvConvexHull2( ptseq, 0, CV_CLOCKWISE, 0 );
            //int hullcount = hull->total;
            
            defects= cvConvexityDefects(ptseq,hull,storage2  );
            CvConvexityDefect* defectArray;
            int j=0;
            CvFont font;
            for(;defects;defects = defects->h_next){
                int nomdef = defects->total;
                if(nomdef == 0)
                    continue;
                defectArray = (CvConvexityDefect*)malloc(sizeof(CvConvexityDefect)*nomdef);
                cvCvtSeqToArray(defects,defectArray, CV_WHOLE_SEQ);
                for(int i=0; i<nomdef; i++){
                    //printf(" defect depth for defect %d %f %d %d\n",i,defectArray[i].depth, defectArray[i].start->x,defectArray[i].start->y);
                    cvLine(imgRealFeed, *(defectArray[i].start), *(defectArray[i].depth_point),CV_RGB(255,255,0),1, CV_AA, 0 );
                    cvCircle( imgRealFeed, *(defectArray[i].depth_point), 5, CV_RGB(0,255,0), 2, 8,0);
                    cvCircle( imgRealFeed, *(defectArray[i].start), 5, CV_RGB(255,0,0), 2, 8,0);
                    cvLine(imgRealFeed, *(defectArray[i].depth_point), *(defectArray[i].end),CV_RGB(255,255,0),1, CV_AA, 0 );
                    //손바닥 중심과 손가락 끝 연결
                    if(i!=2){
                        cvLine(imgRealFeed, center, *(defectArray[i].start),CV_RGB(255,0,0),1, CV_AA, 0);
                        }
                    cvLine(imgRealFeed, center, *(defectArray[i].depth_point),CV_RGB(0,255,0),1, CV_AA, 0);

                }
                char txt[]="0";
                txt[0]='0'+nomdef-1;
                if(nomdef == 6){
                    printf("손가락 엄지손가락과 새끼손가락의 길이:\n %f \n",calcDistance(*defectArray[0].start, *defectArray[5].start));
                    printf("중심점과 각 손가락 사이의 길이:\n %.2f %.2f %.2f %.2f \n",
                           calcDistance(*defectArray[0].depth_point, center),
                           calcDistance(*defectArray[5].depth_point, center),
                           calcDistance(*defectArray[4].depth_point, center),
                           calcDistance(*defectArray[3].depth_point, center));
                    printf("중심점과 각 손가락 끝의 길이:\n %.2f %.2f %.2f %.2f %.2f \n",
                           calcDistance(*defectArray[1].start, center),
                           calcDistance(*defectArray[0].start, center),
                           calcDistance(*defectArray[5].start, center),
                           calcDistance(*defectArray[4].start, center),
                           calcDistance(*defectArray[3].start, center));
                    printf("손목길이 %f \n", calcDistance(*(defectArray[1].depth_point), *(defectArray[2].depth_point)));
                    cvLine(imgRealFeed, *(defectArray[1].depth_point), *(defectArray[2].depth_point),CV_RGB(255,255,0),1, CV_AA, 0 );
                    printf("중심점과 각 손목점과의 거리:\n %f %f\n",
                           calcDistance(*defectArray[1].depth_point, center),
                           calcDistance(*defectArray[2].depth_point, center));
                    

                }
                cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 5, CV_AA);
                cvPutText(imgRealFeed, txt, cvPoint(50, 50), &font, cvScalar(0, 0, 255, 0));
                j++;
                free(defectArray);
            }
            cvReleaseMemStorage( &storage );
            cvReleaseMemStorage( &storage1 );
            cvReleaseMemStorage( &storage2 );
            cvReleaseMemStorage( &storage3 );
        }
    }
}

int main(){
    
    //Mat image = imread("/Users/jeongsooha/MyDesktop/testpictures/5.jpeg");
    Mat image = imread("/Users/jeongsooha/MyDesktop/testpictures/t5_p.JPG");
    cv::resize( image, image, cv::Size( 450, 600), 0, 0, CV_INTER_NN );
    Mat originimage = image.clone();
    //Mat detectedImg;
    IplImage *rawImage = 0, *yuvImage = 0;
    
    Mat mask=getHandMask1(image);
    
    //침식 함수 : 필터 내부의 가장 낮은(어두운)값으로 변환(end연산)
    erode(mask, mask, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2);
    
    double radius;
    
    //손바닥의 중간점을 Point형으로 반환
    Point center=getHandCenter(mask, radius);
    
    //손바닥 중간점을 이용해 손가락 개수를 파악
    cout<<"손바닥 중심점 좌표:"<<center<<", 반지름:"<<radius<<", 손가락 개수"<<getFingerCount(mask, center, radius)<<endl;
    
    yuvImage = new IplImage(mask);
    rawImage = new IplImage(image);
    
    // imshow("Main Data image", mask);
    
    //cvShowImage("Main Data", rawImage);
    //cvShowImage("yuvImage Data", yuvImage);
    
    /***
     *Detect hand features on rawImage
     *IplImage를 이용해서 함수의 인자로 넘기지만 포인터 형이기 때문에
     *IplImage to Mat을 할 필요없이 바로 image가 변한다.
     ***/
    detect(yuvImage,rawImage, center);
    
    //cvShowImage("Add Line on Image", rawImage);
    //imshow("After detect image", image);
    
    //손바닥 중심점 그리기
    circle(image, center, 5, Scalar(255, 0, 0), -1);
    
    //손바닥 영역 그리기
    circle(image, center, (int)(radius*1.5), Scalar(255, 0, 0), 2);
    
    
    //cv::resize( image, image, cv::Size( 450, 600), 0, 0, CV_INTER_NN );
    //cv::resize( mask, mask, cv::Size( 450, 600), 0, 0, CV_INTER_NN );
    //namedWindow( "Original Image With Hand Center", CV_WINDOW_AUTOSIZE );
    //namedWindow( "mask", CV_WINDOW_AUTOSIZE );
    //cvResizeWindow("Original Image With Hand Center", 350, 350);
    //cvResizeWindow("mask", 350, 350);
   
    /*
    imshow("excuted erode function on mask", mask);
    imshow("Original Image", originimage);
    */
    imshow("Detect points on Image", image);

    
    
    waitKey(0);
    
    
    //  }
    
    
}

