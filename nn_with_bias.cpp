#include "nn_with_bias.h"
#include <cmath>
#include "MersenneTwister.h"

using namespace std;

nn_with_bias::nn_with_bias()
{
    values=0;
    weights=0;
    layersizes=0;
    delta_weights=0;
    resetnn();
}

nn_with_bias::nn_with_bias(squashtype a, double b, double m, int c, int* d)
//squash, learnrate, numlayers, layersizes
{
    initializenn( a, b, m, c, d);
}

nn_with_bias::~nn_with_bias()
{
    resetnn();
}

void nn_with_bias::initializenn(squashtype a, double b, double m, int c, int* d)
//squash, learnrate, numlayers, layersizes
{
//delete old data just in case it isn't already done
values=0;
weights=0;
delta_weights=0;
layersizes=0;
resetnn();
//copy over the given valufes
squash= a;
learnrate =b;
momentum =m;
numlayers=c;
layersizes = new int[numlayers+1];
for(int i=0; i<numlayers+1; i++)
{
    layersizes[i]=d[i];
}

//now create the space for the weights and values in the nn and initialize
//the weights at random
    //first initialize the random number generator
MTRand randomNumberGenerator;
values = new double*[numlayers+1];
weights = new double**[numlayers];
delta_weights = new double**[numlayers];
for(int i=0; i<numlayers; i++)
{
    values[i]= new double[layersizes[i]+1];
}
values[numlayers]= new double[layersizes[numlayers]];

for(int i=0; i<numlayers; i++)
{
    weights[i]= new double*[layersizes[i]+1];
    delta_weights[i]= new double*[layersizes[i]+1];
    for(int j=0; j<layersizes[i]+1; j++)
    {
        weights[i][j] = new double[layersizes[i+1]];
        delta_weights[i][j] = new double[layersizes[i+1]];
        for(int k=0; k<layersizes[i+1];k++)
        {
            weights[i][j][k] = randomNumberGenerator.randExc(2)-1;
            delta_weights[i][j][k] = 0;
        }
    }
}
}

void nn_with_bias::resetnn()
//deletes dynamic memory
{
    learnrate=momentum=0;
    if(values !=0)
    {
        for(int i=0; i<numlayers; i++)
        {
            if(values[i]!=0)
            {
                delete values[i];
            }
        }
        delete values;
        values=0;
    }
    if(weights !=0)
    {
        for(int i=0; i<numlayers; i++)
        {
            if(weights[i]!=0)
            {
                delete weights[i];
            }
        }
        delete weights;
        weights=0;
    }
    if(delta_weights !=0)
    {
        for(int i=0; i<numlayers; i++)
        {
            if(delta_weights[i]!=0)
            {
                delete delta_weights[i];
            }
        }
        delete delta_weights;
        delta_weights=0;
    }
    if(layersizes!=0)
        delete layersizes;
    layersizes=0;
    numlayers=0;
}

void nn_with_bias::getLayer(double*& output, int layernumber) const
//input is output- must have space for data
{
    //cout<<"output layer# is "<<layernumber<<endl;
    if(layernumber<=numlayers)
    {
        if(output!=0)
        {
            for(int i=0; i<layersizes[layernumber];i++)
            {
                //cout<<"values["<<layernumber<<"]["<<i<<"]= "<<values[layernumber][i]<<endl;
                output[i]=values[layernumber][i];
                //system("pause");
            }
        }
    }
}

void nn_with_bias::updatenn(double*& input)
// input will be copied- output will be calc
{
    //cout<<"entry layer is of size "<<layersizes[0]<<". Output layer is of size ";
    //cout << getOutputSize()<<" inputs are "<<input[0]<< " and "<<input[1]<<endl;
    for(int i=0; i<layersizes[0];i++)
    {
        values[0][i]= input[i];
    }
    //bias stuff
    for(int i=0; i<numlayers; i++)
        values[i][layersizes[i]]=1;
    for(int i=0; i<numlayers; i++)
    {//for each layer
        for(int j=0; j<layersizes[i+1];j++)//reset the next layer
            values[i+1][j]=0;
        for(int j=0; j<layersizes[i]+1;j++)//for each value in the upstream
        //layer
        {
            for(int k=0; k<layersizes[i+1];k++)//for each weight
            //that contributes to the targeted downrange variable
            {
                values[i+1][k]+=values[i][j]*weights[i][j][k];
            }
        }
        for(int j=0; j<layersizes[i+1];j++)
        {
            //cout<<"values ["<<i+1<<"]["<<j<<"]= "<<values[i+1][j]<<endl;
            values[i+1][j]=squashf(values[i+1][j]);
            //cout<<"values ["<<i+1<<"]["<<j<<"] squashed= "<<values[i+1][j]<<endl;
            //system("pause");
        }
    }
    //cout <<" Run once"<<endl;
}

void nn_with_bias::getOutput(double*& output) const
//input is output- must have space for data
{
    getLayer(output, numlayers);
}

int &nn_with_bias::getLayerSize(int layer) const
//layer#
{
    return layersizes[layer];
}

int &nn_with_bias::getOutputSize() const
//gives size of output layer
{
    return layersizes[numlayers];
}

void nn_with_bias::trainEpoch(double*& target)
//uses the input as target for training for this epoch
{
    //setup memory usage
    double **gradient= new double*[numlayers+1];
    double **error= new double*[numlayers+1];
    double ***delta_weights_temp= new double**[numlayers];
    for(int i=0; i<numlayers; i++)
    {
        delta_weights_temp[i]=new double*[layersizes[i]+1];
        for(int j=0; j<layersizes[i]+1; j++)
        {
            delta_weights_temp[i][j] = new double[layersizes[i+1]];
        }
    }
    for(int i=0; i<numlayers+1; i++)
    {
        gradient[i]= new double[layersizes[i]+1];
        error[i]= new double[layersizes[i]+1];
    }

    //find the error and gradient for the output layer
    for(int i=0; i<layersizes[numlayers]+1; i++)
    {
        //first find the error
        error[numlayers][i]=target[i]-values[numlayers][i];//12:07

        //now find the gradient
        if(squash==hyperbolictangent)
        {
            gradient[numlayers][i]=error[numlayers][i]*(1+values[numlayers][i])*(1-values[numlayers][i]);//
        }
        if(squash==logistic)
        {
            gradient[numlayers][i]=error[numlayers][i]*values[numlayers][i]*(1-values[numlayers][i]);//12:48
        }
    }
    //now find the error and gradient for the hidden layers
    for(int i=numlayers-1; i>0; i--)//references the error layer which is -1 from the equivelent value values layer index
    {
        for(int j=0; j<layersizes[i]+1; j++)//references the values of the error layer
        {
            //find the error
            error[i][j]=0;
            for(int k=0; k<layersizes[i+1]; k++)//references the values of the downstream layer
            {
                error[i][j]+=gradient[i+1][k]*weights[i][j][k];//
            }
            //find the gradient
            gradient[i][j]=0;
            if(squash==hyperbolictangent)
            {
                gradient[i][j]=error[i][j]*(1+values[i][j])*(1-values[i][j]);//
            }
            if(squash==logistic)
            {
                gradient[i][j]=error[i][j]*values[i][j]*(1-values[i][j]);//
            }
        }
    }
    //now we adjust the weights
    for(int i=0; i<numlayers; i++)//for each layer of weight layers
    {
        for(int j=0; j<layersizes[i]; j++)//for each input to that weight layer
        {
            for(int k=0; k<layersizes[i+1]; k++)//for each output
            {
                //adjust weight [i][j][k] and update momentum
                delta_weights_temp[i][j][k]=learnrate*gradient[i+1][k]*values[i][j]+momentum*delta_weights[i][j][k];
                weights[i][j][k]+=delta_weights_temp[i][j][k];//13:06
            }
        }
    }

    //now return memory to the system
    for(int i=0; i<numlayers+1; i++)
    {
        delete gradient[i];
        delete error[i];
    }
    delete gradient;
    delete error;

    if(delta_weights !=0)
    {
        for(int i=0; i<numlayers; i++)
        {
                for(int j=0; j<layersizes[i]+1; j++)
                {
                    delete delta_weights[i][j];
                }
                delete delta_weights[i];
        }
        delete delta_weights;
        delta_weights=0;
    }
    //update the delta_weights
    delta_weights=delta_weights_temp;
    delta_weights_temp=0;
}

double nn_with_bias::squashf(double input)
{
    if(squash==hyperbolictangent)
    {
        return tanh(input);
    }
    if(squash==logistic)
    {
        return 1/(1+pow(2.71828183,-1*input));
    }
    return -1;
}
