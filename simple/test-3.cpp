#include <cstdio>
#include <vector>
#include <random>

using namespace std;

const int n = 2;
const double s = -0.001;
const int inputSize = 3;
const int layerSize[n+1] = {inputSize, 16, 4};
double input[inputSize];


vector<double> a[n+1], b[n+1];
vector<vector<double> > w[n];
vector<double> da[n+1], dbt[n+1];
vector<vector<double> > dwt[n];

double sig(double x) {
    return 1/(1+exp(-x));
}

double dsig(double x) {
    double s=sig(x);
    return s*(1-s);
}

void backprop(int layer) {
    //printf("Beginning backprop w layer %d\n",layer);
    for (int i=0;i<layerSize[layer];++i) {
        //printf("dc/da=%f for neuron %d which has output %f\n",da[layer][i],i,a[layer][i]);
        double dz = da[layer][i]*(a[layer][i]>=0?1:.1);
        da[layer][i]=0;
        //printf("and dc/db=%f\n",dz);
        dbt[layer][i]+=dz;
        if (layer) for (int j=0;j<layerSize[layer-1];++j) {
            //printf("on layer %d\ndc/dw for %d of layer %d to %d of layer %d is %f, where weight is %f\n",layer-1,j,layer-1,i,layer,dz*a[layer-1][j],w[layer-1][i][j]);
            //printf("also dc/da for %d of layer %d from %d of layer %d increases by %f\n",j,layer-1,i,layer,dz*w[layer-1][i][j]);
            da[layer-1][j]+=dz*w[layer-1][i][j];
            dwt[layer-1][i][j]+=dz*a[layer-1][j];
        }
    }
    if (layer) backprop(layer-1);
    //printf("\n");
}

void foreprop(int layer) {
    for (int i=0;i<layerSize[layer];++i) {
        b[layer][i]+=dbt[layer][i]*s;
        dbt[layer][i]=da[layer][i]=0;
        if (layer<n) for (int j=0;j<layerSize[layer+1];++j) {
            w[layer][j][i]+=dwt[layer][j][i]*s;
            dwt[layer][j][i]=0;
        }
    }
    if (layer<n) foreprop(layer+1);
}

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

void populate() {
    for (int i=0;i<=n;++i) {
        if (i<n) w[i].resize(layerSize[i+1]), dwt[i].resize(layerSize[i+1]);
        for (int k=0;k<layerSize[i];++k) {
            a[i].push_back(0);
            b[i].push_back(0);//(rand()*1.0/RAND_MAX)-0.5);
            da[i].push_back(0);
            dbt[i].push_back(0);
            if (i<n) {
                for (int j=0;j<layerSize[i+1];++j) {
                    w[i][j].push_back((rand()*1.0/RAND_MAX)-0.5);
                    dwt[i][j].push_back(0);
                }
            }
        }
    }
}

int main() {
    srand(time(0));
    //freopen("out","w",stdout);
    //get your dataset things in order
    populate();
    //main loop
    unsigned char buf;
    double t = 0;
    int r=0;
    while (1) {
        for (int i=0;i<100;++i) {
            //r=rand()%3;
            double c=0.0;
            for (int j=0;j<inputSize;++j) {
                input[j]=rand()%2;
            }
            buf = input[0]+input[1]+input[2];
            calculate(n);
            for (int j=0;j<layerSize[n];++j) {
                //printf("%f ",a[n][j]);
                t=(buf==j);
                c+=pow(a[n][j]-t,2);
                da[n][j]=(a[n][j]-t);
            }
            if (!i) printf("\nCost: %.10f for %d\n",c,buf);
            backprop(n);
        }
        foreprop(0);
        /*
        for (int i=0;i<=n;++i) {
            for (int j=0;j<layerSize[i];++j) {
                printf("%d %f  ",j,a[i][j]);
            }
            if (i<n) {
                    printf("\n");
                for (int j=0;j<layerSize[i];++j)
                for (int k=0;k<layerSize[i+1];++k) {
                    printf("%d %d %f  ",j,k,w[i][k][j]);
                }
            }
            printf("\n");
        }*/
    }
}