#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

// Define Skin Colors
int Y_MIN = 0;
int Y_MAX = 255;
int Cr_MIN = 133;
int Cr_MAX = 173;
int Cb_MIN = 77;
int Cb_MAX = 127;

// Method to get Skin Tone
Mat getSkin(Mat matCamFeed){
    Mat matPlayerSkin;
    
    // Chaning the Image Format
    cvtColor(matCamFeed,matPlayerSkin,COLOR_BGR2YCrCb);
    
    imshow("YCrCb Color Space", matPlayerSkin);
    
    // Make the test if Color of Skin meets defined color
    inRange(matPlayerSkin,Scalar(Y_MIN,Cr_MIN,Cb_MIN),Scalar(Y_MAX,Cr_MAX,Cb_MAX),matPlayerSkin);
    
    return matPlayerSkin;
}

void  detect(IplImage* imgTonedImage,IplImage* imgRealFeed) {
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
            for(int i = 0; i < maxitem->total; i++ ){
                CvPoint* p = CV_GET_SEQ_ELEM( CvPoint, maxitem, i );
                pt0.x = p->x;
                pt0.y = p->y;
                cvSeqPush( ptseq, &pt0 );
            }
            hull = cvConvexHull2( ptseq, 0, CV_CLOCKWISE, 0 );
            int hullcount = hull->total;
            
            defects= cvConvexityDefects(ptseq,hull,storage2  );
            CvConvexityDefect* defectArray;
            int j=0;
            for(;defects;defects = defects->h_next){
                int nomdef = defects->total;
                if(nomdef == 0)
                    continue;
                defectArray = (CvConvexityDefect*)malloc(sizeof(CvConvexityDefect)*nomdef);
                cvCvtSeqToArray(defects,defectArray, CV_WHOLE_SEQ);
                for(int i=0; i<nomdef; i++){
                    printf(" defect depth for defect %d %f \n",i,defectArray[i].depth);
                    cvLine(imgRealFeed, *(defectArray[i].start), *(defectArray[i].depth_point),CV_RGB(255,255,0),1, CV_AA, 0 );
                    cvCircle( imgRealFeed, *(defectArray[i].depth_point), 5, CV_RGB(0,0,164), 2, 8,0);
                    cvCircle( imgRealFeed, *(defectArray[i].start), 5, CV_RGB(0,0,164), 2, 8,0);
                    cvLine(imgRealFeed, *(defectArray[i].depth_point), *(defectArray[i].end),CV_RGB(255,255,0),1, CV_AA, 0 );
                }
                char txt[]="0";
                txt[0]='0'+nomdef-1;
                CvFont font;
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

int main( int argc, char** argv ){
    
    //
    IplImage *rawImage = 0, *yuvImage = 0, *grayImage=0;
    
    // Setting variable to get Camera Feed
    //VideoCapture objVCapture;
    //objVCapture.open(0);
//    Mat matCamFeed;
//    Mat gray_image;
    Mat gray_image = imread("/Users/jeongsooha/MyDesktop/testpictures/b2.jpeg");
    Mat matCamFeed = imread("/Users/jeongsooha/MyDesktop/testpictures/b2.jpeg");
    // Setting up the Window Size
    //objVCapture.set(CV_CAP_PROP_FRAME_WIDTH,350);
    //objVCapture.set(CV_CAP_PROP_FRAME_HEIGHT, 350);
    
    // Skin Data
    Mat skinMat;
    //testImage.read(gray_image);
    cvtColor(gray_image, gray_image, cv::COLOR_BGR2GRAY);
    grayImage = new IplImage(gray_image);
    
    
    
    // Loop to Capture Feed and Process it
     //while(1){
    
    // Get Feed and Store in Matrix and Display on Screen
    //objVCapture.read(matCamFeed);
    //cvtColor(matCamFeed, gray_image, cv::COLOR_BGR2GRAY);
    // Get Skin Only
    skinMat = getSkin(matCamFeed);
    //threshold( gray_image, skinMat, 127, 255, THRESH_BINARY );
    
    
    // Create Raw Image from Feed
    rawImage = new IplImage(matCamFeed);
    //rawImage = new IplImage(gray_image);
    
    // Create Image for Detected Skin
    yuvImage = new IplImage(skinMat);
    
    // Detect Edges and place on Image
    detect(yuvImage,rawImage);
    
    // Display the Data
    cvShowImage("test Data", grayImage);
    cvShowImage("Main Data", rawImage);
    cvShowImage("yuvImage Data", yuvImage);
    
    waitKey(0);
    
    //  }
    
    
    return 0;
}
