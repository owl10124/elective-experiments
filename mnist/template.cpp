#include <cstdio>
#include <vector>
#include <random>

using namespace std;

const int n = 3;
const double s = -0.0003;
const int inputSize = 784;
const int layerSize[n+1] = {inputSize, 256, 128, 10};
double input[inputSize];

vector<double> a[n+1], z[n+1], b[n+1];
vector<vector<double> > w[n];
vector<double> da[n+1], dbt[n+1];
vector<vector<double> > dwt[n];

void backprop(int layer) {
    //printf("Beginning backprop w layer %d\n",layer);
    for (int i=0;i<layerSize[layer];++i) {
        //printf("dc/da=%f for neuron %d which has output %f\n",da[layer][i],i,a[layer][i]);
        double dz = da[layer][i]*(z[layer][i]>=0?1:.1);
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
        z[layer][i]=b[layer][i];
        if (layer) {
            for (int j=0;j<layerSize[layer-1];++j) {
                //printf("%d %d %d %d\n",layer-1,i,j,w[layer-1][i].size());
                z[layer][i]+=a[layer-1][j]*w[layer-1][i][j];
            }
        } else z[layer][i]=input[i];
        a[layer][i]=max(z[layer][i],0.0);
    }
}

int swap(int x) {
    return ((x&255)<<24)
    + ((x&(255<<8))<<8)
    + ((x&(255<<16))>>8)
    + ((x&(255<<24))>>24);
}

void exportnet(FILE* net) {
    net = fopen("net","w");
    fprintf(net,"%d ",n);
    for (int i=0;i<=n;++i) fprintf(net,"%d ",layerSize[i]);
    fprintf(net,"\n");
    for (int i=0;i<=n;++i) for (int j=0;j<layerSize[i];++j) fprintf(net,"%.10f ",b[i][j]);
    fprintf(net,"\n");
    for (int i=0;i<n;++i) for (int j=0;j<layerSize[i+1];++j) for (int k=0;k<layerSize[i];++k) fprintf(net,"%.10f ",w[i][j][k]);
    fprintf(net,"\n\n");
    fclose(net);
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
            z[i].push_back(0);
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
            z[i].push_back(0);
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

int main() {
    srand(time(0));
    unsigned char buf;
    double t = 0;
    FILE *netin = fopen("net","r");
    //get your dataset things in order


    printf(":)\n");
    //populate(netin);
    populate();
    printf(":(\n");
    fclose(netin);

    FILE *net = fopen("net","w");

    for (int p=0;p<4;++p) {
        FILE *trLbl = fopen("train-labels","rb");
        FILE *trImg = fopen("train-images","rb");
        
        assert(trLbl!=NULL&&!ferror(trLbl));
        assert(trImg!=NULL&&!ferror(trImg));

        int x;
        fread(&x,4,1,trLbl); x=swap(x); assert(x==2049);
        fread(&x,4,1,trImg); x=swap(x); assert(x==2051);

        fread(&x,4,1,trLbl);
        fread(&x,4,1,trImg);
        fread(&x,4,1,trImg);
        fread(&x,4,1,trImg);

        //main loop
        for (int k=0;k<600;++k) {
            for (int i=0;i<100;++i) {
                //int t = k*10+i;
                double c=0.0;
                for (int j=0;j<layerSize[0];++j) {
                    fread(&buf,1,1,trImg);
                    input[j]=buf/255.0;
                }
                fread(&buf,1,1,trLbl);

                //printf("%d\n",buf);
                calculate(n);
                for (int j=0;j<10;++j) {
                    t=(buf==j);
                    if (!i) printf("%f %f  ",a[n][j],t);
                    c+=pow(a[n][j]-t,2);
                    da[n][j]=(a[n][j]-t);
                }
                if (!i) printf("\n%.10f %d %d %d %d\n\n",c,buf,p,k,i);
                backprop(n);
            }
            foreprop(0);
            if (!(k%16)) exportnet(net);
        }
    }
}
