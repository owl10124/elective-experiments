#include <cstdio>
#include <vector>
#include <cmath>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

using namespace std;

const int n = 3;
const int inputSize = 28*28;
const int layerSize[n+1] = {inputSize, 256, 128, 10};
double input[inputSize];

vector<double> a[n+1], b[n+1];
vector<vector<double> > w[n];
vector<double> da[n+1], dbt[n+1];
vector<vector<double> > dwt[n];

void calculate(int layer) {
    if (layer) calculate(layer-1);
    for (int i=0;i<layerSize[layer];++i) {
        a[layer][i]=b[layer][i];
        if (layer) {
            for (int j=0;j<layerSize[layer-1];++j) {
                //printf("%d %d %d %d\n",layer-1,i,j,w[layer-1][i].size());
                a[layer][i]+=a[layer-1][j]*w[layer-1][i][j];
            }
        } else a[layer][i]=input[i];
        a[layer][i]=max(a[layer][i],0.0);
    }
}

int swap(int x) {
    return ((x&255)<<24)
    + ((x&(255<<8))<<8)
    + ((x&(255<<16))>>8)
    + ((x&(255<<24))>>24);
}

void populate(FILE* net) {
    int bint; double bdouble;
    fscanf(net,"%d",&bint), assert(bint==n);
    for (int i=0;i<=n;++i) fscanf(net,"%d",&bint), assert(layerSize[i]==bint);
    for (int i=0;i<=n;++i) {
        if (i<n) w[i].resize(layerSize[i+1]), dwt[i].resize(layerSize[i+1]);
        for (int k=0;k<layerSize[i];++k) {
            fscanf(net,"%lf",&bdouble);
            a[i].push_back(0);
            b[i].push_back(bdouble);//(rand()*1.0/RAND_MAX)-0.5);
            da[i].push_back(0);
            dbt[i].push_back(0);
        }
    }
    for (int i=0;i<n;++i) for (int j=0;j<layerSize[i+1];++j) for (int k=0;k<layerSize[i];++k) fscanf(net,"%lf",&bdouble), w[i][j].push_back(bdouble), dwt[i][j].push_back(0);
}

int main(int argc, char* argv[]) {
    assert(argc>=3);
    FILE *netin = fopen(argv[1],"r");
    populate(netin);
    fclose(netin);
    //get your dataset things in order

    int i = 1;
    while (++i<argc) {
    cv::Mat img = cv::imread(argv[i],cv::IMREAD_GRAYSCALE);
    cv::Mat dst;
    cv::Size size(28,28);

    cv::resize(img,dst,size);
    unsigned char buf = 0, *p;


    //main loop
    for (int k=0;k<28;++k) {
        p = dst.ptr<unsigned char>(k);
        for (int j=0;j<28;++j) {
            input[28*k+j]=1-(p[j]/255.0);
        }
    }

    calculate(n);
    int m = -1; double c = -INFINITY;
    for (int j=0;j<10;++j) {
        printf("Weight for %d: %.6f\n",j,a[n][j]);
        if (a[n][j]>c) m=j,c=a[n][j];
    }
    printf("========================\nFinal prediction for %s: %d\n\n",argv[i],m);
    }
}
