#include <cstdio>
#include <vector>
#include <random>

using namespace std;

const int n = 3;
const int inputSize = 784;
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
                    w[i][j].push_back(((rand()*1.0/RAND_MAX)-0.5)*0.1);
                    dwt[i][j].push_back(0);
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    srand(time(0));
    FILE *netin = fopen(argv[1],"r");
    //freopen("out","w",stdout);
    //get your dataset things in order

    int total = 0, correct = 0;
    unsigned char buf = 0;

    populate(netin);
    fclose(netin);

    FILE *tsLbl = fopen("test-labels","rb");
    FILE *tsImg = fopen("test-images","rb");
    
    assert(tsLbl!=NULL&&!ferror(tsLbl));
    assert(tsImg!=NULL&&!ferror(tsImg));

    int x;
    fread(&x,4,1,tsLbl); x=swap(x); assert(x==2049);
    fread(&x,4,1,tsImg); x=swap(x); assert(x==2051);

    fread(&x,4,1,tsLbl);
    fread(&x,4,1,tsImg);
    fread(&x,4,1,tsImg);
    fread(&x,4,1,tsImg);

    //main loop
    for (int k=0;k<10000;++k) {
        for (int j=0;j<layerSize[0];++j) {
            fread(&buf,1,1,tsImg);
            input[j]=buf/255.0;
        }
        fread(&buf,1,1,tsLbl);

        //printf("%d\n",buf);
        calculate(n);
        int m = -1; double c = -INFINITY;
        for (int j=0;j<10;++j) {
            if (a[n][j]>c) m=j,c=a[n][j];
        }
        if (m==buf) ++correct;
        ++total;
    }
    printf("accuracy = %lf; correct = %d; total = %d\n",(correct*1.0)/total,correct,total);
}
