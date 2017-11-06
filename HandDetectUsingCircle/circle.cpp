//
//  main.cpp
//  HandDetectUsingCircle
//
//  Created by 하정수 on 12/08/2017.
//  Copyright © 2017 하정수. All rights reserved.
//

#include <iostream>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

using namespace std;
using namespace cv;

#define SCALE 1.5

// Method to get Skin Tone
Mat getHandMask(const Mat& image){
    //컬러 공간 변환 BGR->YCrCb
    int Y_MIN = 0;
    int Y_MAX = 255;
    int Cr_MIN = 133;
    int Cr_MAX = 173;
    int Cb_MIN = 77;
    int Cb_MAX = 127;
    Mat YCrCb;
    
    // 이미지 포맷 변환
    cvtColor(image, YCrCb, COLOR_BGR2YCrCb);
    
    // 피부색이 정의된 색의 범위에 포함되는지 확인하고 포함되면 0으로 나머지는 1로 변환
    inRange(YCrCb,Scalar(Y_MIN,Cr_MIN,Cb_MIN),Scalar(Y_MAX,Cr_MAX,Cb_MAX),YCrCb);
    
    return YCrCb;
    
}

//CvPoint와 CvPoint의 거리 계산
double calcDistance(const CvPoint& p1,const CvPoint& p2){
    double p1x = p1.x, p1y = p1.y, p2x = p2.x, p2y = p2.y;
    return sqrt(pow(p1x - p2x,2) + pow(p1y - p2y,2));
}

//CvPoint와 Point의 거리 계산
double calcDistance(const CvPoint& p1,const Point& p2){
    double p1x = p1.x, p1y = p1.y, p2x = p2.x, p2y = p2.y;
    return sqrt(pow(p1x - p2x,2) + pow(p1y - p2y,2));
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

//손가락의 개수를 파악해서 리턴
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


//특징점이 될만한 점들을 찾는 함수
String detect(IplImage* imgTonedImage,IplImage* imgRealFeed, const Point& center) {
    String info ="";
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
            char txt[]="0";
            for(;defects;defects = defects->h_next)
            {
                int nomdef = defects->total;
                //printf("nomdef = defects->total : %d\n", nomdef);
                if(nomdef == 0)
                    continue;
                defectArray = (CvConvexityDefect*)malloc(sizeof(CvConvexityDefect)*nomdef);
                //printf("sizeof(defectArray)/sizeof(*defectArray) : %lu\n",(int)sizeof(defectArray)/sizeof(*defectArray));
                cvCvtSeqToArray(defects,defectArray, CV_WHOLE_SEQ);
                for(int i=0; i<nomdef; i++){
                    txt[0]='0'+i;
                    //printf(" defect depth for defect %d %f %d %d\n",i,defectArray[i].depth, defectArray[i].start->x,defectArray[i].start->y);
                    cvLine(imgRealFeed, *(defectArray[i].start), *(defectArray[i].depth_point),CV_RGB(255,255,0),1, CV_AA, 0 );
                    cvCircle( imgRealFeed, *(defectArray[i].depth_point), 5, CV_RGB(0,255,0), 2, 8,0);
                    cvCircle( imgRealFeed, *(defectArray[i].start), 5, CV_RGB(255,0,0), 2, 8,0);
                    cvLine(imgRealFeed, *(defectArray[i].depth_point), *(defectArray[i].end),CV_RGB(255,255,0),1, CV_AA, 0 );
                    if(i!=5){
                        //손가락이 5개가 아닐때 i가 손가락의 갯수보다 하나 작은 수까지 돌아야한다. *(defectArray[i+1].start 때문에
                        if(nomdef!=6 && nomdef>i){
                            cvLine(imgRealFeed, *(defectArray[i].start), *(defectArray[nomdef-1].start),CV_RGB(200,255,200),1, CV_AA, 0 );
                        }
                        else{
                            cvLine(imgRealFeed, *(defectArray[i].start), *(defectArray[i+1].start),CV_RGB(200,255,200),1, CV_AA, 0 );
                            printf("손 끝과 끝 %d %d %f\n",i, i+1, calcDistance(*(defectArray[i].start), *(defectArray[i+1].start)));
                        }
                    }
                    else{
                        //손가락이 5개이면 마지막 점과 첫번째 선을 연결
                        cvLine(imgRealFeed, *(defectArray[5].start), *(defectArray[0].start),CV_RGB(200,255,200),1, CV_AA, 0 );
                        printf("손 끝과 끝 %d %d %f\n",5, 0, calcDistance(*(defectArray[5].start), *(defectArray[0].start)));
                    }
                    
                    //손바닥 중심과 손가락 끝 연결
                    cvLine(imgRealFeed, center, *(defectArray[i].start),CV_RGB(255,0,0),1, CV_AA, 0);
                    cvLine(imgRealFeed, center, *(defectArray[i].depth_point),CV_RGB(0,255,0),1, CV_AA, 0);
                    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.7, 0.7, 0, 2, CV_AA);
                    cvPutText(imgRealFeed, txt, *(defectArray[i].depth_point), &font, cvScalar(0, 255, 0, 0));
                    cvPutText(imgRealFeed, txt, *(defectArray[i].start), &font, cvScalar(0, 0, 255, 0));
                    //printf("%d x= %d y= %d \n", i,  defectArray[i].start->x,defectArray[i].start->y);
                }
                
                //txt[0]='0'+nomdef-1;
                
                
                //손가락이 5개일때만 적용
                if(nomdef == 6){
                    printf("%d %d \n",defectArray[1].depth_point->x,defectArray[1].depth_point->y);
                    printf("엄지손가락과 새끼손가락의 길이:\n %f \n",calcDistance(*(defectArray[1].start), *(defectArray[3].start)));
                    cvLine(imgRealFeed, *(defectArray[1].start), *(defectArray[3].start),CV_RGB(160,160,160),1, CV_AA, 0 );
                    printf("중심점과 각 손가락 사이의 길이:\n %.2f %.2f %.2f %.2f \n",
                           calcDistance(*(defectArray[0].depth_point), center),
                           calcDistance(*(defectArray[5].depth_point), center),
                           calcDistance(*(defectArray[4].depth_point), center),
                           calcDistance(*(defectArray[3].depth_point), center));
                    printf("중심점과 각 손가락 끝의 길이:\n %.2f %.2f %.2f %.2f %.2f \n",
                           calcDistance(*(defectArray[1].start), center),
                           calcDistance(*(defectArray[0].start), center),
                           calcDistance(*(defectArray[5].start), center),
                           calcDistance(*(defectArray[4].start), center),
                           calcDistance(*(defectArray[3].start), center));
                    printf("손목길이 %f \n", calcDistance(*(defectArray[1].depth_point), *(defectArray[2].depth_point)));
                    cvLine(imgRealFeed, *(defectArray[1].depth_point), *(defectArray[2].depth_point),CV_RGB(255,255,0),1, CV_AA, 0 );
                    printf("중심 점과 각 손목 점과의 거리:\n %f %f\n",
                           calcDistance(*(defectArray[1].depth_point), center),
                           calcDistance(*(defectArray[2].depth_point), center));
                    
                    info = "중심 점과 각 손가락 사이의 길이 : "+
                    to_string(calcDistance(*(defectArray[0].depth_point), center))+" |\t"+
                    to_string(calcDistance(*(defectArray[5].depth_point), center))+" |\t"+
                    to_string(calcDistance(*(defectArray[4].depth_point), center))+" |\t"+
                    to_string(calcDistance(*(defectArray[3].depth_point), center))+"\n"+
                    "손목길이 : "+
                    to_string(calcDistance(*(defectArray[1].depth_point), *(defectArray[2].depth_point)))+"\n"+
                    "중심 점과 각 손목 점과의 거리 : "+
                    to_string(calcDistance(*(defectArray[1].depth_point), center))+" |\t"+
                    to_string(calcDistance(*(defectArray[2].depth_point), center))+"\n";
                    
                    
                }
                
                j++;
                free(defectArray);
            }
            cvReleaseMemStorage( &storage );
            cvReleaseMemStorage( &storage1 );
            cvReleaseMemStorage( &storage2 );
            cvReleaseMemStorage( &storage3 );
        }
    }
    return info;
}

//텍스트 파일에 저장
void writeFile(String savePath, String info){
    cout << "writeFile..." <<endl;
    ofstream output;
    output.open(savePath);
    output << info << endl;
    output.close();
}

int main(){
    String info="";
    //이미지가 저장되어 있는 PATH
    String filePath = "./images/";
    //txt파일을 저장할 PATH
    String savePath = "./";
    
    //불러올 이미지 이름
    String imgName = "KB5";
    
    //테스트할 이미지
    //Xcode에서는 PATH 설정에 유의
    Mat image = imread(filePath+imgName+".jpg");
    //txt를 저장할 PATH 설정
    savePath = savePath+imgName+".txt";
    
    //이미지 크기 변환
    cv::resize( image, image, cv::Size( 450, 600), 0, 0, CV_INTER_NN );
    
    Mat originimage = image.clone();
    IplImage *rawImage = 0, *yuvImage = 0;
    
    //Mask이미지 저장
    Mat mask=getHandMask(image);
    
    //침식 함수 : 필터 내부의 가장 낮은(어두운)값으로 변환(end연산)
    erode(mask, mask, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2);
    
    double radius;
    
    //손바닥의 중간점을 Point형으로 반환
    Point center=getHandCenter(mask, radius);
    info = "손바닥 중심점 좌표 : "+to_string(center.x)+", "+to_string(center.y)+"\n";
    info = info + "반지름 : "+to_string(radius)+"\n";
    //손바닥 중간점을 이용해 손가락 개수를 파악
    int fingernum = getFingerCount(mask, center, radius,SCALE);
    
    cout<<"손바닥 중심점 좌표:"<<center<<", 반지름:"<<radius<<", 손가락 개수"<<fingernum<<endl;
    
    yuvImage = new IplImage(mask);
    rawImage = new IplImage(image);
    
    /***
     * Detect hand features on rawImage
     * IplImage를 이용해서 함수의 인자로 넘기지만 포인터 형이기 때문에
     * IplImage to Mat을 할 필요없이 바로 image가 변한다.
     ***/
    info = info + detect(yuvImage,rawImage, center);
    
    //손가락 갯수를 창에 표시
    char txt[]="0";
    CvFont font;
    txt[0]='0'+fingernum;
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 5, CV_AA);
    cvPutText(rawImage, txt, cvPoint(50, 50), &font, cvScalar(0, 0, 255, 0));

    info = info +"손가락 개수 : "+ to_string(fingernum);
    
    //손바닥 중심점 그리기
    circle(image, center, 5, Scalar(255, 0, 0), -1);
    
    //손바닥 영역 그리기
    circle(image, center, (int)(radius*SCALE), Scalar(255, 0, 0), 2);
    
    //이미지에 대한 정보가 테스트 파일에 저장
    writeFile(savePath, info);
    
    imshow("Detect points on Image", image);

    waitKey(0);
    
}

