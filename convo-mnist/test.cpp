#include <cstdio>
#include <vector>
#include <random>

using namespace std;

const int n = 2; //The number of layers -1. Halved. So there should actually be 0 to 6
const int hf = 2, f = 2*hf+1, m = 2; //Feature size. hf is just half of it lol. Pooling size is 2.
const int imgSize = 28;
const int inputSize = 1, outputSize = 10;
const int layerSize[n+1] = {inputSize,4,8};//this happens between 0s and 1s. so after 0, 2, and 4.
double input[imgSize*imgSize];

int dim(int i) {return i?((dim(i-1)-f)/m+1):imgSize;} //Size of nth input layer.

vector<vector<double> > a[2*n+1]; //a is neuron value. mod 2: 0 means input (pooled and *not* ReLUed), 1 means post-convo. a[layer][item][pixel]
vector<vector<double> > w[n][f*f]; //w is feature. w[layer][pixel][target][source]

const int nn = 2;
const int nLayerSize[nn+1] = {layerSize[n]*dim(n)*dim(n),100,outputSize};

vector<double> na[nn+1], nz[nn+1], nb[nn+1];
vector<vector<double> > nw[nn];

void calculate(int layer) { //Updates all 'z's and 'a's according to the new input. Goes from 0 to 2*n+nn.
    if (layer) calculate(layer-1);
    //printf("\n%d\n",layer);
    if (layer<=2*n) {
        if (layer%2) { //yay convo
            int d = dim(layer/2);
            for (int i=0;i<layerSize[layer/2+1];++i) {
                for (int p=0;p<(d-f+1)*(d-f+1);++p) {
                    a[layer][i][p]=0;
                    for (int k=0;k<f*f;++k) for (int j=0;j<layerSize[layer/2];++j) {
                        int nx = p/(d-f+1)+k/f, ny = p%(d-f+1)+k%f;
                        a[layer][i][p]+=max(a[layer-1][j][nx*d+ny],0.0)*w[layer/2][k][i][j];
                    }
                    //printf("%03.2f ",a[layer][i][p]);
                }
                //printf("\n");
            }
        } else { //maxpool
            int d = dim(layer/2);
            for (int i=0;i<layerSize[layer/2];++i) {
                for (int p=0;p<d*d;++p) {
                    //if (!(p%d)) printf("\n");
                    if (layer) { //get max and prop to it
                        int pd=dim(layer/2-1)-f+1; //dimension of pre-pool
                        double n = -INFINITY;
                        for (int j=m*(p/d);j<min(m*(p/d+1),pd);++j) for (int k=m*(p%d);k<min(m*(p%d+1),pd);++k) {
                            if (a[layer-1][i][j*pd+k]>n) n=a[layer-1][i][j*pd+k];
                        }
                        assert(n!=-INFINITY);
                        a[layer][i][p]=n;
                    } else a[layer][i][p]=input[p];
                    //printf("%03.2f ",a[layer][i][p]);
                }
                //printf("\n");
            }
        }
    }

    if (layer>=2*n) {
        int l = layer-2*n;
        for (int i=0;i<nLayerSize[l];++i) {
            nz[l][i]=nb[l][i];
            if (l) {
                for (int j=0;j<nLayerSize[l-1];++j) {
                    nz[l][i]+=na[l-1][j]*nw[l-1][i][j];
                }
            } else nz[l][i]=a[2*n][i%layerSize[n]][i/layerSize[n]];
            na[l][i]=max(nz[l][i],0.0);
            //if (!l) printf("%03.2f ",na[l][i]);
        }
        //printf("\n");
    } 
    //printf("\n\ncalculated\n\n\n\n");
}

void importnet(FILE* net) { //Imports the network from a file.
    int bint; double bdouble;
    fscanf(net,"%d",&bint), assert(bint==n);
    fscanf(net,"%d",&bint), assert(bint==nn);
    fscanf(net,"%d",&bint), assert(bint==imgSize);
    fscanf(net,"%d",&bint), assert(bint==f);
    fscanf(net,"%d",&bint), assert(bint==m);
    for (int i=0;i<n;++i) fscanf(net,"%d",&bint), assert(layerSize[i]==bint);
    for (int i=0;i<=nn;++i) fscanf(net,"%d",&bint), assert(nLayerSize[i]==bint);
    printf("a\n");
    for (int i=0;i<=2*n;++i) {
        int d = (dim(i/2)-(i%2?f-1:0));
        printf("Layer %d has dimension %d and count %d\n",i,d,layerSize[(i+1)/2]);
        for (int j=0;j<layerSize[(i+1)/2];++j) {
            a[i].push_back(vector<double>());
            for (int p=0;p<d*d;++p){
                a[i][j].push_back(0);
            }
        }
    }
    printf("b\n");
    for (int i=0;i<n;++i) {
        for (int p=0;p<f*f;++p) {
            for (int j=0;j<layerSize[i+1];++j) {
                w[i][p].push_back(vector<double>());
                for (int k=0;k<layerSize[i];++k) {
                    fscanf(net,"%lf",&bdouble), w[i][p][j].push_back(bdouble);
                }
            }
        }
    }
    printf("c\n");
    for (int i=0;i<=nn;++i) {
        if (i<nn) nw[i].resize(nLayerSize[i+1]);
        for (int j=0;j<nLayerSize[i];++j) {
            fscanf(net,"%lf",&bdouble), nb[i].push_back(bdouble);
            na[i].push_back(0);
            nz[i].push_back(0);
        }
    }
    printf("d\n");
    for (int i=0;i<nn;++i) for (int j=0;j<nLayerSize[i+1];++j) for (int k=0;k<nLayerSize[i];++k) {
        fscanf(net,"%lf",&bdouble), nw[i][j].push_back(bdouble);
    }
    printf("imported!\n");
}

int swap(int x) {
    return ((x&255)<<24)
    + ((x&(255<<8))<<8)
    + ((x&(255<<16))>>8)
    + ((x&(255<<24))>>24);
}

int main(int argc, char *argv[]) {
    //freopen("out","w",stdout);
    FILE *netin = fopen(argv[1],"r");
    //get your dataset things in order

    importnet(netin);
    fclose(netin);

    int total = 0, correct = 0;
    unsigned char buf = 0;
    
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
        for (int j=0;j<imgSize*imgSize;++j) {
            fread(&buf,1,1,tsImg);
            input[j]=buf/255.0;
        }
        fread(&buf,1,1,tsLbl);

        //printf("%d\n",buf);
        calculate(2*n+nn);
        int m = -1; double c = -INFINITY;
        for (int j=0;j<10;++j) {
            if (na[nn][j]>c) m=j,c=na[nn][j];
        }
        if (m==buf) ++correct;
        ++total;
        //printf("%d %d\naccuracy = %lf; correct = %d; total = %d\n\n",m,buf,(correct*1.0)/total,correct,total);
    }
    printf("\naccuracy = %lf; correct = %d; total = %d\n",(correct*1.0)/total,correct,total);
}
