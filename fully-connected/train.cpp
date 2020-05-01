#include <cstdio>
#include <vector>
#include <random>

using namespace std;

const int batch = 100; //The number of samples for one gradient descent step.
const double s = -0.03*batch; //This is the step size / learning rate. Take a value between -0.1 and -0.01.

const int n = 3; //The number of layers -1.
const int inputSize = 784, outputSize = 10;
const int layerSize[n+1] = {inputSize, 256, 128, outputSize};
double input[inputSize];

vector<double> a[n+1], z[n+1], b[n+1]; //a is neuron value, z is value pre-ReLU, and b is bias (constant to be added).
vector<vector<double> > w[n]; //w is edge weight.
vector<double> da[n+1], dbt[n+1]; //the t represents 'total'. da is not preserved across calculations.
vector<vector<double> > dwt[n];

void backprop(int layer) { //Recursive backpropagation. Goes from n to 0.
    for (int i=0;i<layerSize[layer];++i) { //Goes through each node in the current layer,
        double dz = da[layer][i]*(z[layer][i]>=0?1:.1); //calculates leaky ReLU (dz/dC),
        da[layer][i]=0; //Resets da of current layer.
        dbt[layer][i]+=dz; //Updates d(bias).
        if (layer) for (int j=0;j<layerSize[layer-1];++j) {
            da[layer-1][j]+=dz*w[layer-1][i][j]; //Updates d(previous neurons).
            dwt[layer-1][i][j]+=dz*a[layer-1][j]; //Updates d(weight).
        }
    }
    if (layer) backprop(layer-1);
}

void foreprop(int layer) { //The actual gradient descent step. Clears all d* variables. Goes from 0 to n.
    for (int i=0;i<layerSize[layer];++i) {
        b[layer][i]+=dbt[layer][i]*s; //Updates biases.
        dbt[layer][i]=da[layer][i]=0;
        if (layer<n) for (int j=0;j<layerSize[layer+1];++j) {
            w[layer][j][i]+=dwt[layer][j][i]*s; //Updates weights.
            dwt[layer][j][i]=0; 
        }
    }
    if (layer<n) foreprop(layer+1);
}

void calculate(int layer) { //Updates all 'z's and 'a's according to the new input. Goes from 0 to n.
    if (layer) calculate(layer-1);
    for (int i=0;i<layerSize[layer];++i) {
        z[layer][i]=b[layer][i];
        if (layer) {
            for (int j=0;j<layerSize[layer-1];++j) {
                z[layer][i]+=a[layer-1][j]*w[layer-1][i][j];
            }
        } else z[layer][i]=input[i];
        a[layer][i]=max(z[layer][i],0.0);
    }
}

void exportnet(FILE* net) { //Exports the network to a file. Prints n, # neurons per layer, then all neuron biases, then all edge weights.
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

void importnet(FILE* net) { //Imports the network from a file.
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

void initialise() { //Initialises the network if the importing fails.
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
    
    //Import network from file.
    FILE *netin = fopen("net","r");
    importnet(netin);
    fclose(netin);

    FILE *net = fopen("net","w"); //For export.

    //Get your dataset in order.

    for (int p=0;p<128;++p) {
        //Open the dataset and do what you want with it.

        //Main loop.
        for (int k=0;k<600;++k) { //Set this number according to your dataset.
            for (int i=0;i<100;++i) {
                double c=0.0;
                for (int j=0;j<inputSize;++j) {
                    //Populate with inputs.
                }
                calculate(n);

                for (int j=0;j<outputSize;++j) {
                    t=0;//Populate with outputs.
                    c+=pow(a[n][j]-t,2); //Cost function.
                    da[n][j]=(a[n][j]-t); //da/dc.
                }
                backprop(n);
            }
            foreprop(0);
            exportnet(net); //Obviously, don't do this every iteration.
        }
    }
}
