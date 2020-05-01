#include <cstdio>
#include <vector>
#include <random>

using namespace std;

const double s = -0.0003; //This is the step size / learning rate. Take a value between -0.1 and -0.01.
const double ns = -0.0003;

const int n = 2; //The number of layers -1. Halved. So there should actually be 0 to 6
const int hf = 2, f = 2*hf+1, m = 2; //Feature size. hf is just half of it lol. Pooling size is 2.
const int imgSize = 28;
const int inputSize = 1, outputSize = 10;
const int layerSize[n+1] = {inputSize,4,8};//this happens between 0s and 1s. so after 0, 2, and 4.
double input[imgSize*imgSize];

int dim(int i) {return i?((dim(i-1)-f)/m+1):imgSize;} //Size of nth input layer.
double clip(double x) {return x<0?0:x>8?8:x;}
bool clipped(double x) {return x<0||x>8;}

vector<vector<double> > a[2*n+1]; //a is neuron value. mod 2: 0 means input (pooled and *not* ReLUed), 1 means post-convo. a[layer][item][pixel]
vector<vector<double> > w[n][f*f]; //w is feature. w[layer][pixel][target][source]
vector<vector<double> > da[2*n+1]; //the t represents 'total'. da is not preserved across calculations.
vector<vector<double> > dwt[n][f*f];

const int nn = 2;
const int nLayerSize[nn+1] = {layerSize[n]*dim(n)*dim(n),256,outputSize};

vector<double> na[nn+1], nz[nn+1], nb[nn+1];
vector<vector<double> > nw[nn];
vector<double> nda[nn+1], ndbt[nn+1];
vector<vector<double> > ndwt[nn];

void backprop(int layer) { //Recursive backpropagation. Goes from 2n+nn to 0.
    //printf("Beginning backprop w layer %d\n",layer);
    if (layer>=2*n) {
        int l = layer-2*n;
        for (int i=0;i<nLayerSize[l];++i) {
            //printf("dc/da=%f for neuron %d which has output %f\n",nda[l][i],i,na[l][i]);
            double ndz = nda[l][i]*(clipped(nz[l][i])?1:.05);
            nda[l][i]=0;
            //printf("and dc/db=%f\n",ndz);
            ndbt[l][i]+=ndz;
            if (l) for (int j=0;j<nLayerSize[l-1];++j) {
                //printf("on layer %d\ndc/dw for %d of layer %d to %d of layer %d is %f, where weight is %f\n",l-1,j,l-1,i,l,ndz*na[l-1][j],nw[l-1][i][j]);
                //printf("also dc/da for %d of layer %d from %d of layer %d increases by %f\n",j,l-1,i,l,ndz*nw[l-1][i][j]);
                nda[l-1][j]+=ndz*nw[l-1][i][j];
                ndwt[l-1][i][j]+=ndz*na[l-1][j];
                //printf("%d %f %f\n",l,nda[l-1][j],ndwt[l-1][i][j]);
            } else da[2*n][i%layerSize[n]][i/layerSize[n]]=ndz;
        }
    }
    if (layer<=2*n) {
        if (layer%2) { //convo layer, prop backwards thru 'ReLU then convo'
            int d = dim(layer/2); //well yes. this is the _previous_ one
            for (int i=0;i<layerSize[layer/2+1];++i) { //Goes through each node in the current layer
                for (int p=0;p<(d-f+1)*(d-f+1);++p) { 
                    double dz = da[layer][i][p];
                    if (layer&&dz) for (int k=0;k<f*f;++k) for (int j=0;j<layerSize[layer/2];++j) {
                        int x = p/(d-f+1)+k/f, y = p%(d-f+1)+k%f;
                        assert(x<d&&y<d);
                        da[layer-1][j][x*d+y]+=2*dz*w[layer/2][k][i][j]*(clipped(a[layer-1][j][x*d+y])?1:.05); //Updates d(previous neurons).
                        dwt[layer/2][k][i][j]+=2*dz*a[layer-1][j][x*d+y]; //Updates d(weight).
                        //printf("%d %d %f %f\n",layer,x*d+y,da[layer-1][j][x*d+y],dwt[layer/2][k][i][j]);
                    }
                    da[layer][i][p]=0; //Resets da of current layer. 
                }
            }
        } else { //it's an input layer, and you need to propagate through a maxpool
            int d = dim(layer/2);
            for (int i=0;i<layerSize[layer/2];++i) {
                for (int p=0;p<d*d;++p) { //from small layer
                    if (layer) { //get max and prop to it
                        int ind=-1, pd=dim(layer/2-1)-f+1; double n = -INFINITY;
                        //printf("%d %d\n",p,pd);
                        for (int j=m*(p/d);j<min(m*(p/d+1),pd);++j) for (int k=m*(p%d);k<min(m*(p%d+1),pd);++k) {
                            if (a[layer-1][i][j*pd+k]>=n) n=a[layer-1][i][j*pd+k], ind=j*pd+k; //save max if applicable
                        }
                        assert(ind+1);
                        da[layer-1][i][ind]=da[layer][i][p];
                        //printf("%d %d %f\n",layer,ind,da[layer-1][i][ind]);
                    }
                    da[layer][i][p]=0;
                }
            }
        }
    }
    if (layer) backprop(layer-1);
}

void foreprop(int layer) { //The actual gradient descent step. Clears all d* variables. Goes from 0 to n+nn. This is intentional.
    if (layer) foreprop(layer-1);
    //printf("fore %d\n",layer);
    if (layer>=n) {
        int l = layer-n;
        for (int i=0;i<nLayerSize[l];++i) {
            nb[l][i]+=ndbt[l][i]*ns;
            ndbt[l][i]=0;
            if (l<nn) for (int j=0;j<nLayerSize[l+1];++j) {
                nw[l][j][i]+=ndwt[l][j][i]*ns;
                ndwt[l][j][i]=0;
            }
        }
    } else { //only foreprop convo layers. or rather weights. this is stupid.
        for (int i=0;i<layerSize[layer+1];++i) {
            for (int k=0;k<f*f;++k) for (int j=0;j<layerSize[layer];++j) {
                //printf("%f \n",dwt[layer][k][i][j]*s);
                w[layer][k][i][j]+=dwt[layer][k][i][j]*s; //Updates weights.
                dwt[layer][k][i][j]=0; 
            }
        }
    }
}

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
                        a[layer][i][p]+=clip(a[layer-1][j][nx*d+ny])*w[layer/2][k][i][j];
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
            na[l][i]=clip(nz[l][i]);
            //if (!l) printf("%03.2f ",na[l][i]);
        }
        //printf("\n");
    } 
    //printf("\n\ncalculated\n\n\n\n");
}

void exportnet(FILE* net) { //Exports the network to a file. Prints n, nn, # neurons per layer, then all edge weights and biases.
    fprintf(net,"%d %d %d %d %d\n",n,nn,imgSize,f,m);
    for (int i=0;i<n;++i) fprintf(net,"%d ",layerSize[i]);
    for (int i=0;i<=nn;++i) fprintf(net,"%d ",nLayerSize[i]);
    fprintf(net,"\n");
    for (int i=0;i<n;++i) for (int p=0;p<f*f;++p) for (int j=0;j<layerSize[i+1];++j) for (int k=0;k<layerSize[i];++k) fprintf(net,"%.10f ",w[i][p][j][k]);
    fprintf(net,"\n");
    for (int i=0;i<=nn;++i) for (int j=0;j<nLayerSize[i];++j) fprintf(net,"%.10f ",nb[i][j]);
    fprintf(net,"\n");
    for (int i=0;i<nn;++i) for (int j=0;j<nLayerSize[i+1];++j) for (int k=0;k<nLayerSize[i];++k) fprintf(net,"%.10f ",nw[i][j][k]);
    fprintf(net,"\n\n");
    fclose(net);
}

void importnet(FILE* net) { //Imports the network from a file.
    int bint; double bdouble;
    if (net) {
        fscanf(net,"%d",&bint), assert(bint==n);
        fscanf(net,"%d",&bint), assert(bint==nn);
        fscanf(net,"%d",&bint), assert(bint==imgSize);
        fscanf(net,"%d",&bint), assert(bint==f);
        fscanf(net,"%d",&bint), assert(bint==m);
        for (int i=0;i<n;++i) fscanf(net,"%d",&bint), assert(layerSize[i]==bint);
        for (int i=0;i<=nn;++i) fscanf(net,"%d",&bint), assert(nLayerSize[i]==bint);
    }
    printf("a\n");
    for (int i=0;i<=2*n;++i) {
        int d = (dim(i/2)-(i%2?f-1:0));
        printf("Layer %d has dimension %d and count %d\n",i,d,layerSize[(i+1)/2]);
        for (int j=0;j<layerSize[(i+1)/2];++j) {
            a[i].push_back(vector<double>());
            da[i].push_back(vector<double>());
            for (int p=0;p<d*d;++p){
                a[i][j].push_back(0);
                da[i][j].push_back(0);
            }
        }
    }
    printf("b\n");
    for (int i=0;i<n;++i) {
        for (int p=0;p<f*f;++p) {
            for (int j=0;j<layerSize[i+1];++j) {
                w[i][p].push_back(vector<double>());
                dwt[i][p].push_back(vector<double>());
                for (int k=0;k<layerSize[i];++k) {
                    if (net) fscanf(net,"%lf",&bdouble), w[i][p][j].push_back(bdouble); else w[i][p][j].push_back(((rand()*1.0/RAND_MAX)-0.5)*2); 
                    dwt[i][p][j].push_back(0);
                }
            }
        }
    }
    printf("c\n");
    for (int i=0;i<=nn;++i) {
        if (i<nn) nw[i].resize(nLayerSize[i+1]), ndwt[i].resize(nLayerSize[i+1]);
        for (int j=0;j<nLayerSize[i];++j) {
            if (net) fscanf(net,"%lf",&bdouble), nb[i].push_back(bdouble); else nb[i].push_back(0);
            na[i].push_back(0);
            nz[i].push_back(0);
            nda[i].push_back(0);
            ndbt[i].push_back(0);
        }
    }
    printf("d\n");
    for (int i=0;i<nn;++i) for (int j=0;j<nLayerSize[i+1];++j) for (int k=0;k<nLayerSize[i];++k) {
        if (net) fscanf(net,"%lf",&bdouble), nw[i][j].push_back(bdouble); 
        else nw[i][j].push_back(((rand()*1.0/RAND_MAX)-0.5)*0.1); 
        ndwt[i][j].push_back(0);
    }
    printf("imported!\n");
}

int swap(int x) {
    return ((x&255)<<24)
    + ((x&(255<<8))<<8)
    + ((x&(255<<16))>>8)
    + ((x&(255<<24))>>24);
}

int main() {
    //freopen("out","w",stdout);
    srand(time(0));
    unsigned char buf;
    double t = 0;
    FILE *netin = fopen("net","r");
    //get your dataset things in order

    printf(":)\n");
    //importnet(netin);
    importnet(nullptr);
    printf(":(\n");
    fclose(netin);

    FILE *net;
    FILE *trLbl = fopen("train-labels","rb");
    FILE *trImg = fopen("train-images","rb");
    assert(trLbl!=NULL&&!ferror(trLbl));
    assert(trImg!=NULL&&!ferror(trImg));

    for (int p=0;p<4;++p) {
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
                for (int j=0;j<imgSize*imgSize;++j) {
                    fread(&buf,1,1,trImg);
                    input[j]=buf/255.0;
                }
                fread(&buf,1,1,trLbl);

                //printf("%d\n",buf);
                calculate(2*n+nn);
                for (int j=0;j<10;++j) {
                    t=(buf==j);
                    if (!i) printf("%f %f  ",na[nn][j],t);
                    c+=pow(na[nn][j]-t,2);
                    nda[nn][j]=(na[nn][j]-t);
                }
                if (!i) printf("\n%.10f %d %d %d %d\n\n",c,buf,p,k,i);
                backprop(2*n+nn);
            }
            foreprop(n+nn);
            if (!(k%15)) net=fopen("net","w"), exportnet(net), fclose(net);
        }
        rewind(trLbl);
        rewind(trImg);
    }
}
