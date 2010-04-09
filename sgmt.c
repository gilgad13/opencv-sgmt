#include "cv.h"
#include "highgui.h"
#include "stdio.h"

//! Mouse callback
IplImage *frame;
IplImage *marker;
void on_mouse( int event, int x, int y, int flags, void* param );
void EqualizeHist(IplImage* img);
int GrabPointsFromMask(IplImage* mask, CvPoint** points,  int max_points);
int FormFeatrueMat(IplImage* img, IplImage* mask, CvMat** out, CvMat* avg, int max_points);
void calcMahonobis(IplImage* img, IplImage* output, const CvMat* avg, const CvMat* covar);

int main( int argc, char* argv[])
{
    CvCapture* capture;

    if(argc == 1)
        capture = cvCreateCameraCapture(0);
    else
        capture = cvCreateFileCapture( argv[1] );

    cvNamedWindow("Video", CV_WINDOW_AUTOSIZE);

    // App. the first frame grabbed is trash, so grab two
    frame = cvQueryFrame(capture);
    frame = cvQueryFrame(capture);
    printf("IS_IMAGE: %d\n", CV_IS_IMAGE(frame));
    marker = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);

    // Create mask
    cvZero(marker); 
    cvShowImage("Video", frame);
    cvSetMouseCallback("Video", on_mouse, NULL);
    while(cvWaitKey(33) == -1);
    cvSetMouseCallback("Video", NULL, NULL);

    CvMat* features[100000];
    CvMat* covar = cvCreateMat(3, 3, CV_32FC1);
    CvMat* avg = cvCreateMat(1, 3, CV_32FC1); 
    int feature_count = FormFeatrueMat(frame, marker, features, avg, 100000);
    if(feature_count == 0) {
        printf("No region selected.  Quitting\n");
        exit(1);
    }
    cvCalcCovarMatrix((const CvArr**)features, feature_count, covar, NULL, CV_COVAR_NORMAL | CV_COVAR_SCALE);
    printf("Avg is [%f | %f | %f]\n", avg->data.fl[0],avg->data.fl[1],avg->data.fl[2]);
    cvInvert(covar, covar, CV_SVD_SYM);
    IplImage* mah = cvCreateImage(cvGetSize(frame), IPL_DEPTH_32F, 1);
    IplImage* disp = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 3);
    calcMahonobis(frame, mah, avg, covar);

    CvScalar mean, stddev; 
    while(1)
    {
        // Grab new frame
        frame = cvQueryFrame(capture);

        // Convert to HSV and split into channels
        /*        cvCvtColor(frame, frame, CV_RGB2HSV); */
        /*        EqualizeHist(frame);*/
        /*        FormFeatrueMat(frame, marker, features, avg, 100000);*/
        /*        cvCalcCovarMatrix((const CvArr**)features, feature_count, covar, NULL, CV_COVAR_NORMAL | CV_COVAR_SCALE);*/
        /*        cvInvert(covar, covar, CV_SVD_SYM);*/
        calcMahonobis(frame, mah, avg, covar);

        /*        cvConvertScale(mah, disp, 1, 0);*/
        printf("Avg is [%f | %f | %f]\n", avg->data.fl[0],avg->data.fl[1],avg->data.fl[2]); 
        /*        cvShowImage("Video", mah);*/
        if(cvWaitKey(33) == 27) break;
    }
    return 0;
}

void on_mouse( int event, int x, int y, int flags, void* param )
{
    static int button_down = 0;
    static int last_x = -1;
    static int last_y = -1;


    if(button_down) {
        cvLine(marker, cvPoint(last_x, last_y), cvPoint(x, y), cvScalar(0xFF, 0, 0, 0), 2, 8, 0); 
        last_x = x;
        last_y = y;
        IplImage *disp = cvClone(frame);
        cvSubS(frame, CV_RGB(255, 255, 255), disp, marker);
        cvShowImage("Video", disp);
    }

    if(event == CV_EVENT_LBUTTONDOWN) {
        button_down = 1;
        last_x = x;
        last_y = y;
    } else if(event == CV_EVENT_LBUTTONUP) {
        button_down = 0;
    }

}

void EqualizeHist(IplImage* img)
{
    IplImage* c1 = cvCreateImage(cvGetSize(img), img->depth, 1); 
    IplImage* c2 = cvCreateImage(cvGetSize(img), img->depth, 1); 
    IplImage* c3 = cvCreateImage(cvGetSize(img), img->depth, 1); 

    cvSplit(img, c1, c2, c3, 0);
    cvEqualizeHist(c1, c1);
    cvEqualizeHist(c2, c2);
    cvEqualizeHist(c3, c3);
    cvMerge(c1, c2, c3, NULL, img);

    return;
}

int GrabPointsFromMask(IplImage* mask, CvPoint** points,  int max_points)
{
    int index = 0;
    for(int y = 0; (y < mask->height) && (index < max_points); y++) {
        uchar* pmask = (uchar*) (mask->imageData + y * mask->widthStep);
        for(int x = 0; (x < mask->width) && (index < max_points); x++) {
            if(pmask[x] != 0) {
                points[index] = (CvPoint*)malloc(sizeof(CvPoint));
                *(points[index++]) = cvPoint(x, y);
            }
        }
    }

    return index;
}

int FormFeatrueMat(IplImage* img, IplImage* mask, CvMat** out, CvMat* avg, int max_points)
{
    int index = 0;

    /* Alright, the general plot here is to:
     * 1) Pull out the RGB values of img corresponding to non-zero points in
     *      mask
     * 2) Append these 3 chars to the bytestream pointed to by mat_data
     * 3) Construct a Matrix headder that will allow us to interpret these
     *      points as a p-row by 3-column matrix
     */

    // Grab the points we need from the mask
    for(int y = 0; (y < img->height) && (index < max_points); y++) {
        uchar* pimg = (uchar*) (img->imageData + y * img->widthStep);
        uchar* pmask = (uchar*) (mask->imageData + y * mask->widthStep);
        for(int x = 0; (x < img->width) && (index < max_points); x++) {
            if(pmask[x] != 0) {
                CvScalar pixel = cvGet2D(img, y, x);
                out[index] = cvCreateMat(3, 1, CV_8UC1);
                out[index]->data.ptr[0] = pixel.val[0];
                out[index]->data.ptr[1] = pixel.val[1];
                out[index]->data.ptr[2] = pixel.val[2];
                index++;
                avg->data.fl[0] = avg->data.fl[0]*(1.0 - 1.0/index) + (1.0/index)*pixel.val[0]; 
                avg->data.fl[1] = avg->data.fl[1]*(1.0 - 1.0/index) + (1.0/index)*pixel.val[1]; 
                avg->data.fl[2] = avg->data.fl[2]*(1.0 - 1.0/index) + (1.0/index)*pixel.val[2]; 
                /*                printf("Caught(%d) [%f %f %f]\n", (index / 3), pixel.val[0], pixel.val[1], pixel.val[2]);*/
            }
        } 
    }

    return index;
}

void calcMahonobis(IplImage* img, IplImage* output, const CvMat* avg, const CvMat* covar)
{
    cvZero(output);
    IplImage* thresh = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    CvMat* pixel = cvCreateMat(1, 3, CV_32FC1);
    for(int y = 0; y < img->height; y++) {
        uchar *iptr = (uchar*)(img->imageData + y * img->widthStep);
        double *optr = (double*)(output->imageData + y * output->widthStep);
        uchar *threshptr = (uchar*)(thresh->imageData + y * thresh->widthStep);
        for(int x = 0; x < img->width; x++) {
            cvmSet(pixel, 0, 0, iptr[3*x]);
            cvmSet(pixel, 0, 1, iptr[3*x+1]);
            cvmSet(pixel, 0, 2, iptr[3*x+2]);
            optr[x] = cvMahalanobis(avg, pixel, covar);
            threshptr[x] = (optr[x] < 10) ? 0xFF : 0;
            /*            printf("[%f | %f | %f] and [%f | %f | %f] = [%f]\n", cvmGet(avg, 0, 0), cvmGet(avg, 0, 1), cvmGet(avg, 0, 2),*/
            /*                    cvmGet(pixel, 0, 0), cvmGet(pixel, 0, 1), cvmGet(pixel, 0, 2),*/
            /*                    optr[x]);*/
        }
    }
    cvShowImage("Video", thresh);
    if(cvWaitKey(33) == 27);
    return;
}
