#ifndef NN_WITH_BIAS_H
#define NN_WITH_BIAS_H
#include <iostream>
#include <cstdlib>

using namespace std;
enum squashtype {hyperbolictangent, logistic};
class nn_with_bias
{
public:
    nn_with_bias();
    nn_with_bias(squashtype, double, double, int, int*);
    //squash, learnrate, momentum, numlayers, layersizes
    ~nn_with_bias();
    void initializenn(squashtype, double, double, int, int*);
    //squash, learnrate, momentum, numlayers, layersizes
    void resetnn();//deletes dynamic memory
    void getLayer(double*&, int) const;//input is output- must have space for data
    void updatenn(double*&);// input will be copied- output will be calc
    void getOutput(double*&) const;//input is output- must have space for data
    int& getLayerSize(int) const;//layer#
    int& getOutputSize() const;//gives size of output layer
    void trainEpoch(double*&);//uses the input layer as target for training
    //for this epoch

public:
    double squashf(double);

    int* layersizes;
    double** values;
    double*** weights;
    double*** delta_weights;
    squashtype squash;
    double momentum;
    double learnrate;
    int numlayers;
};

#endif // NN_WITH_BIAS_H
