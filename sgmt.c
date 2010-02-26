#include "cv.h"
#include "highgui.h"
#include "stdio.h"

//! Mouse callback
IplImage *frame;
IplImage *marker;
void on_mouse( int event, int x, int y, int flags, void* param );
void EqualizeHist(IplImage* img);
void FormFeatrueMat(IplImage* img, IplImage* mask, CvMat* out);

int main( int argc, char* argv[])
{
    CvCapture* capture;

    if(argc == 1)
        capture = cvCreateCameraCapture(0);
    else
        capture = cvCreateFileCapture( argv[1] );

    cvNamedWindow("Video", CV_WINDOW_AUTOSIZE);

    frame = cvQueryFrame(capture);
    marker = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);

    // Create mask
    cvZero(marker); 
    cvShowImage("Video", marker);
    cvSetMouseCallback("Video", on_mouse, NULL);
    while(cvWaitKey(33) == -1);
    cvSetMouseCallback("Video", NULL, NULL);

    CvMat* features;
    FormFeatrueMat(frame, marker, features);
    CvScalar mean, stddev;

    while(1)
    {
        // Grab new frame
        frame = cvQueryFrame(capture);
        
        // Convert to HSV and split into channels
        cvCvtColor(frame, frame, CV_RGB2HSV); 
/*        EqualizeHist(frame);*/
        cvShowImage("Video", frame);
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

void FormFeatrueMat(IplImage* img, IplImage* mask, CvMat* out)
{
    // Because I don't want to allocate 3*width*height chars every time, have a
    // practical limit on the number of points we'll sample:
    const int MAX_POINTS = 1000;

    if(out)
        cvReleaseMat(&out);
    out = (CvMat*) malloc(sizeof(CvMat));

    uchar* mat_data = malloc(MAX_POINTS*3);
    int mat_off = 0;

    if(!out || !mat_data)
        exit(66);

    /* Alright, the general plot here is to:
     * 1) Pull out the RGB values of img corresponding to non-zero points in
     *      mask
     * 2) Append these 3 chars to the bytestream pointed to by mat_data
     * 3) Construct a Matrix headder that will allow us to interpret these
     *      points as a p-row by 3-column matrix
     */

    // Grab the points we need from the mask
    for(int y = 0; y < img->height && (mat_off/3 < MAX_POINTS); y++) {
        uchar* pimg = (uchar*) (img->imageData + y * img->widthStep);
        uchar* pmask = (uchar*) (mask->imageData + y * mask->widthStep);
        for(int x = 0; (x < img->width) && (mat_off/3 < MAX_POINTS); x++) {
            if(pmask[x] != 0) {
                CvScalar pixel = cvGet2D(img, x, y);
                mat_data[mat_off++] = pixel.val[0];
                mat_data[mat_off++] = pixel.val[1];
                mat_data[mat_off++] = pixel.val[2];
/*                printf("Caught(%d) [%d %d %d]\n", (mat_off / 3), pixel.val[0], pixel.val[1], pixel.val[2]);*/
            }
        }

    }
    
    // If mat_off is 0 we caught nothing, return
    if(mat_off == 0)
        return;

    cvInitMatHeader(out, 3, mat_off/3, CV_8UC1, mat_data, CV_AUTOSTEP); 

    return;
}
