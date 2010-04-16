/*
 * This program will segment an image using mahalanobis distance as a metric. At
 * the first screen, use the mouse to select pixels which will act as the
 * "training set".  Then hit enter (on a linux system, anyway).
 *
 * execute with no commandline arguments to use webcam, or a filename to load
 * that video.
 *
 * Author: Adam Brockett
 */

#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <sys/time.h>

// Threshold for Mahalanobis distance, currently found empirically
#define THRESHOLD 1
#define FEAT_MAX 10000
IplImage *frame;
IplImage *marker;

int GrabPointsFromMask(IplImage* mask, CvPoint** points,  int max_points);
void GetFeatures(IplImage* img, CvPoint** point_list, int num_points, CvMat* covar, CvMat* avg);
void calcMahalanobis(const IplImage* img, IplImage* output, const CvMat* avg, const CvMat* covar);
void EqualizeHist(IplImage* img);
// Mouse callback
void on_mouse( int event, int x, int y, int flags, void* param );

int main( int argc, char* argv[])
{
    CvCapture* capture;

    if(argc == 1)
        capture = cvCreateCameraCapture(0);
    else
        capture = cvCreateFileCapture( argv[1] );

    cvNamedWindow("Video", CV_WINDOW_AUTOSIZE);

    // Apparently the first frame grabbed is trash, so grab two
    frame = cvQueryFrame(capture);
    frame = cvQueryFrame(capture);

    // Create mask
    marker = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
    cvZero(marker); 
    cvShowImage("Video", frame);
    cvSetMouseCallback("Video", on_mouse, NULL);
    while(cvWaitKey(33) != 10);
    cvSetMouseCallback("Video", NULL, NULL);

    // Pull out the features from that mask
    CvPoint* point_list[FEAT_MAX];
    int point_count = GrabPointsFromMask(marker, point_list, FEAT_MAX); 
    if(point_count == 0) {
        printf("No region selected.  Quitting\n");
        exit(1);
    }
    CvMat* covar = cvCreateMat(3, 3, CV_32FC1);
    CvMat* avg = cvCreateMat(1, 3, CV_32FC1); 
    GetFeatures(frame, point_list, point_count, covar, avg);
    printf("Avg is [%f | %f | %f]\n", avg->data.fl[0],avg->data.fl[1],avg->data.fl[2]);

    // Form the inverse covariance matrix needed for Mahalanobis distance calculation
    cvInvert(covar, covar, CV_SVD_SYM); 
    IplImage* mah = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);

    // Timing code
    struct timeval beg, end;
    double elapsed_time;
    while(1)
    {
        // Grab new frame
        frame = cvQueryFrame(capture);

        gettimeofday(&beg, NULL);

        calcMahalanobis(frame, mah, avg, covar);
        gettimeofday(&end, NULL);

        // compute and print the elapsed time in millisec
        elapsed_time = (end.tv_sec - beg.tv_sec) * 1000.0;      // sec to ms
        elapsed_time += (end.tv_usec - beg.tv_usec) / 1000.0;   // us to ms

        printf("Time %f ms ", elapsed_time);
/*        printf("Avg is [%f | %f | %f]", avg->data.fl[0],avg->data.fl[1],avg->data.fl[2]); */
        printf("\n");

        cvShowImage("Video", mah);
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

/* 
 * Parameters:
 *  - img           input image to pull pixels from
 *  - point_list    array of points to pull pixels from
 *  - num_points    number of points in above array 
 *  - covar         (return) covariance between the 3 channels (preallocated to be 3 by 3, CV_32FC1
 *  - avg           (return) average for each of the channels (preallocated to be 1 by 3, CV_32FC1
 */
void GetFeatures(IplImage* img, CvPoint** point_list, int num_points, CvMat* covar, CvMat* avg)
{
    int index = 0;

    /* Alright, the general plot here is to:
     * 1) Pull out the RGB values of img according as the list of points
     * 2) Create a new 3x1 matrix, fill it with the RGB values of the points,
     *      whilst simultaneously calculating the running average
     * 3) Add a pointer to this matrix to the features array.
     * 4) Feed this features array into the cvCalcCovarMatrix function to
     *      calculate the covariance of the RGB channels in the pixels named in
     *      point_list
     */

    CvMat* features[num_points];
    for(int i = 0; i < num_points; i++) {
        CvScalar pixel = cvGet2D(img, point_list[i]->y, point_list[i]->y);
        features[index] = cvCreateMat(3, 1, CV_8UC1);
        features[index]->data.ptr[0] = pixel.val[0];
        features[index]->data.ptr[1] = pixel.val[1];
        features[index]->data.ptr[2] = pixel.val[2];
        index++;
        avg->data.fl[0] = avg->data.fl[0]*(1.0 - 1.0/index) + (1.0/index)*pixel.val[0]; 
        avg->data.fl[1] = avg->data.fl[1]*(1.0 - 1.0/index) + (1.0/index)*pixel.val[1]; 
        avg->data.fl[2] = avg->data.fl[2]*(1.0 - 1.0/index) + (1.0/index)*pixel.val[2]; 
    }
    cvCalcCovarMatrix((const CvArr**)features, num_points, covar, NULL, CV_COVAR_NORMAL | CV_COVAR_SCALE); 

}

/*
 * Parameters:
 *  - img       input image to be segmented
 *  - output    1 channel, IPL_DEPTH_8U image holding result of thresholding on mahalanobis
 *  - avg       1 by 3 matrix holding average values for each channel
 *  - covar     3 by 3 matrix holding *inverted* covariance matrix for the 3 channels
 */
void calcMahalanobis(const IplImage* img, IplImage* output, const CvMat* avg, const CvMat* covar)
{
    CvMat* pixel = cvCreateMat(1, 3, CV_32FC1);
    for(int y = 0; y < img->height; y++) {
        uchar *iptr = (uchar*)(img->imageData + y * img->widthStep);
        uchar *optr = (uchar*)(output->imageData + y * output->widthStep);
        for(int x = 0; x < img->width; x++) {
            cvmSet(pixel, 0, 0, iptr[3*x]);
            cvmSet(pixel, 0, 1, iptr[3*x+1]);
            cvmSet(pixel, 0, 2, iptr[3*x+2]);
            optr[x] = (cvMahalanobis(avg, pixel, covar) < 10) ? 0xFF : 0;
        }
    }
    return;
}

