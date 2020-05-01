#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <algorithm>
#include "raytrace.hpp"
using namespace std;
using namespace cv;

const int cameraWidth=96, cameraHeight=96;
double scale=0.005, x=0, y=-4, z=1, angle=0;


int main() {
    Mat img = Mat(cameraWidth,cameraHeight,CV_8UC3,{255,255,255});
    vector<Tri*> triarray;

    double length = 1.5, l=-3, r=3;
    for (double i=0;i<5;++i) {
        triarray.push_back(new Poly({{i*length,-i,r},{i*length,1-i,r},{i*length,1-i,l},{i*length,-i,l}}));
        triarray.push_back(new Poly({{i*length,-i,r},{(i+1)*length,-i,r},{(i+1)*length,-i,l},{i*length,-i,l}}));
    }

    char key;
    while (key-27) {
        for (int i=0;i<cameraHeight;++i) for (int j=0;j<cameraWidth;++j) {
            Ray ray = Ray({x-0.1,y-0.02,z},{x,y+(i-cameraHeight/2)*scale,z+(j-cameraWidth/2)*scale});
            double x = INFINITY;
            for (auto k: triarray) x=min(x,k->intersect(ray));
            Vec3b v = img.at<Vec3b>(i,j);
            v[2]=255*(x!=INFINITY);
            v[1]=64;
            v[0]=(int)(x*32)%180;
            img.at<Vec3b>(i,j)=v;
        }
        //imwrite("fox1.png",img,{IMWRITE_PNG_COMPRESSION});
        cvtColor(img,img,COLOR_HSV2BGR);
        imshow("Testing",img);
        key = waitKey();
        switch (key) {
            case 0: y-=0.1; break;
            case 1: y+=0.1; break;
            case 2: z-=0.1; break;
            case 3: z+=0.1; break;
            case 'z': x+=0.1; break;
            case 'c': x-=0.1; break;
        }
    }
}
