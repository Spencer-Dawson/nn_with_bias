#include <QtCore/QCoreApplication>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include "nn_with_bias.h"
#include "MersenneTwister.h"

#define TRAININGSETSIZE 1000
#define OUTPUTHIGH 1
#define OUTPUTLOW -1
#define NUMBEROFEPOCHS 10000
//#define SQUASHTYPE hyperbolictangent;

using namespace std;
bool isClassA(double*&);
bool isClassB(double*&);
bool isClassC(double*&);
void fillset(double**&,int,MTRand*&);//set,set_size,random#generator
void fillItem(double*&,MTRand*&);//set,item_number,random#generator
int test(double*&, double&, double&, double&);//test one item
int train(double*&, double&, double&, double&);//train one item
void testTrain(double**&,int,MTRand*&);//conducts the expiriment
double testTrainOnce(double**&,int,MTRand*&);//conducts the expiriment
double testOnce(double**&,int,double&, double&, double&,MTRand*&);
//double testOnce(double**&,int,MTRand*&);
double trainOnce(double**&,int,double&, double&, double&,MTRand*&);
void swap(double*&, double*&);

nn_with_bias* nn;
int main(int argc, char *argv[])
{
    cout<<"Program started."<<endl;
    //set up variables
    double** training_set;
    training_set = new double*[TRAININGSETSIZE];
    for(int i=0; i<TRAININGSETSIZE; i++)
        training_set[i]= new double[5];

    MTRand* random_number_generator;
    random_number_generator = new MTRand;

    int layersizes[3] = {5, 30, 3};
    nn= new nn_with_bias;
    nn->initializenn(hyperbolictangent, .00002, .9, 2, layersizes);

    fillset(training_set, TRAININGSETSIZE, random_number_generator);
    cout<<"Data substantiated"<<endl;

    //run test
    //cout<<"running test";
    //double error1=0, error2=0, error3=0, error=0;
    //cout<<"running test "<<endl;
    //error= testOnce(training_set, TRAININGSETSIZE, error1, error2, error3, random_number_generator);
    //cout <<"error1 is "<<error1<<" error2 is "<<error2<<" error3 is "<<error3<<" error is "<<error<<endl;

    //run training
    testTrain(training_set, TRAININGSETSIZE, random_number_generator);
    cout<<"expiriment finished"<<endl;

    //return system memory
    for(int i=0; i<TRAININGSETSIZE; i++)
        delete training_set[i];
    delete training_set;

    delete random_number_generator;

    delete nn;
    system("pause");
    return EXIT_SUCCESS;
}

bool isClassA(double*& input)
{
    double sum=0;
    for(int i=0; i<5; i++)
        sum+=input[i];
    if(sum>=3)
        return true;
    return false;
}
bool isClassB(double*& input)
{
    double sum=0;
    for(int i=0; i<5; i++)
        sum+=input[i];
    if(sum<3)
    {
        if(sum>2)
            return true;
    }
    return false;
}
bool isClassC(double*& input)
{
    double sum=0;
    for(int i=0; i<5; i++)
        sum+=input[i];
    if(sum<=2)
        return true;
    return false;
}

void fillset(double**& set,int set_size, MTRand*& random_number_generator)
//set,set_size,random#generator
{
    for(int i=0; i<set_size; i++)
    {
        fillItem(set[i], random_number_generator);
        //if(isClassA(set[i]))
            //cout<<"Is class A"<<endl;
        //if(isClassB(set[i]))
            //cout<<"Is class B"<<endl;
        //if(isClassC(set[i]))
            //cout<<"Is class C"<<endl;
        //system("pause");
    }
}

void fillItem(double*& item, MTRand*& random_number_generator)
//set,item_number,random#generator
{
    static int a_b_or_c=0;//used to make sure population is correctly distributed

    if(a_b_or_c==0)
    {
        do
        {
            for(int i=0; i<5; i++)
                item[i]=random_number_generator->rand(1);
        }
    while(!isClassA(item));
    }
    if(a_b_or_c==1)
    {
        do
        {
            for(int i=0; i<5; i++)
                item[i]=random_number_generator->rand(1);
        }
    while(!isClassB(item));
    }
    if(a_b_or_c==2)
    {
        do
        {
            for(int i=0; i<5; i++)
                item[i]=random_number_generator->rand(1);
        }
    while(!isClassC(item));
    }

    a_b_or_c++;
    if(a_b_or_c>=3)
        a_b_or_c=0;
    //cout<<item[0]<<" "<<item[1]<<" "<<item[2]<<" "<<item[3]<<" "<<item[4]<<endl;
    //if(isClassA(item))
        //cout<<"Is class A"<<endl;
    //if(isClassB(item))
        //cout<<"Is class B"<<endl;
    //if(isClassC(item))
        //cout<<"Is class C"<<endl;
}

int test(double*& input, double& error1, double& error2, double& error3)
//tests input and returns it's classification
//returns it's error by reference
{
    //cout<<"Test Started"<<endl;
    //first do the forward pass and get the test results
    double* nn_output = new double[3];
    //cout<<"about to try forward pass"<<endl;
    nn->updatenn(input);
    //cout<<"forward pass acheived getting output"<<endl;
    nn->getOutput(nn_output);
    //cout<<"output received"<<endl;
    //cout <<"NN input is "<<input[0]<<" "<<input[1]<<" "<<input[2]<<" "<<input[3]<<" "<<input[4]<<endl;
    //cout<<"output is "<<nn_output[0]<<" "<<nn_output[1]<<" "<<nn_output[2]<<endl;
    int highest=0;
    if(nn_output[1]>nn_output[0])
        highest=1;
    if(nn_output[2]>nn_output[highest])
        highest=2;

    //now find the error
    error1=error2=error3=0;
    if(highest==0)
        error1+=(OUTPUTHIGH-nn_output[0])*(OUTPUTHIGH-nn_output[0]);
    else
        error1+=(OUTPUTLOW-nn_output[0])*(OUTPUTLOW-nn_output[0]);
    if(highest==1)
        error2+=(OUTPUTHIGH-nn_output[1])*(OUTPUTHIGH-nn_output[1]);
    else
        error2+=(OUTPUTLOW-nn_output[1])*(OUTPUTLOW-nn_output[1]);
    if(highest==2)
        error3+=(OUTPUTHIGH-nn_output[2])*(OUTPUTHIGH-nn_output[2]);
    else
        error3+=(OUTPUTLOW-nn_output[2])*(OUTPUTLOW-nn_output[2]);

    //now cleanup and return the test result
    delete nn_output;
    return highest;
}

int train(double*& input, double& error1, double& error2, double& error3)
//tests input and returns it's classification
//but only after training the network
{
    //first do the forward pass and get the test results
    double* nn_output = new double[3];
    nn->updatenn(input);
    nn->getOutput(nn_output);
    int highest=0;
    if(nn_output[1]>nn_output[0])
        highest=1;
    if(nn_output[2]>nn_output[highest])
        highest=2;

    //now find the error
    if(highest==0)
        error1=(OUTPUTHIGH-nn_output[0])*(OUTPUTHIGH-nn_output[0]);
    else
        error1=(OUTPUTLOW-nn_output[0])*(OUTPUTLOW-nn_output[0]);
    if(highest==1)
        error1=(OUTPUTHIGH-nn_output[1])*(OUTPUTHIGH-nn_output[1]);
    else
        error1=(OUTPUTLOW-nn_output[1])*(OUTPUTLOW-nn_output[1]);
    if(highest==2)
        error1=(OUTPUTHIGH-nn_output[2])*(OUTPUTHIGH-nn_output[2]);
    else
        error1=(OUTPUTLOW-nn_output[2])*(OUTPUTLOW-nn_output[2]);

    //now train the network
    if(isClassA(input))
    {
        nn_output[0]=OUTPUTHIGH;
        nn_output[1]=OUTPUTLOW;
        nn_output[2]=OUTPUTLOW;
    }
    if(isClassB(input))
    {
        nn_output[0]=OUTPUTLOW;
        nn_output[1]=OUTPUTHIGH;
        nn_output[2]=OUTPUTLOW;
    }
    if(isClassC(input))
    {
        nn_output[0]=OUTPUTLOW;
        nn_output[1]=OUTPUTLOW;
        nn_output[2]=OUTPUTHIGH;
    }
    nn->trainEpoch(nn_output);

    //now cleanup and return the test result
    delete nn_output;
    return highest;
}

void testTrain(double**& set, int set_size, MTRand*& random_number_generator)
{
    cout<<"All testing done before backpropogation."<<endl;
    for(int i=0; i<100; i++)
    {
        cout<<"running test "<<i+1<<endl;
        testTrainOnce(set,set_size,random_number_generator);
        //cout<<"test "<<i+1<<" completed"<<endl;
    }
    system("pause");
    for(int i=0; i<900; i++)
    {
        cout<<"running test "<<i+101<<endl;
        testTrainOnce(set,set_size,random_number_generator);
        //cout<<"test "<<i+1<<" completed"<<endl;
    }
    system("pause");
    for(int i=0; i<NUMBEROFEPOCHS; i++)
    {
        cout<<"running test "<<i+1001<<endl;
        testTrainOnce(set,set_size,random_number_generator);
        //cout<<"test "<<i+1<<" completed"<<endl;
    }
    system("pause");
}

double testTrainOnce(double**& set, int set_size, MTRand*& random_number_generator)
{
    //now test the set
    int numcorrect=0;
    double error1=0, error2=0, error3=0, error=0;
    double temp_error1, temp_error2, temp_error3;
    //cout<<"running test "<<endl;
    error= testOnce(set, set_size, error1, error2, error3, random_number_generator);
    //cout<<"test completed"<<endl;
    //cout<<"running training"<<endl;
    trainOnce(set, set_size, temp_error1, temp_error2, temp_error3, random_number_generator);
    //cout<<"test completed"<<endl;
    cout <<"error1 is "<<error1<<" error2 is "<<error2<<" error3 is "<<error3<<" error is "<<error<<endl;
    //system("pause");
    return error;
}
double testOnce(double**& set, int set_size, double& error1, double& error2, double& error3, MTRand*& random_number_generator)
{
    //now test the set
    //cout<<"running testOnce"<<endl;
    int numcorrect=0, nn_classification;
    double error=0;
    double temp_error1, temp_error2, temp_error3;
    //cout<<"running testOnce- 1"<<endl;
    for(int i=0; i<set_size; i++)
    {
        //cout<<"running test on"<<i<<endl;
        nn_classification=test(set[i], temp_error1, temp_error2, temp_error3);
        int actual_classification;
        if(isClassA(set[i]))
            actual_classification=0;
        if(isClassB(set[i]))
            actual_classification=1;
        if(isClassC(set[i]))
            actual_classification=2;
        if(nn_classification==actual_classification)
            numcorrect++;
        error1+=temp_error1;
        error2+=temp_error2;
        error3+=temp_error3;
    }
    cout<<numcorrect<<" out of "<<set_size<<". "<<100*numcorrect/set_size<<"% identified correctly."<<endl;
    error=error1+error2+error3;
    return error;
}

double trainOnce(double**& set,int set_size,double& error1, double& error2, double& error3,MTRand*& random_number_generator)
{
    //first randomize the set
    for(int i=0; i<4*set_size; i++)
        swap(set[random_number_generator->randInt(set_size-1)],set[random_number_generator->randInt(set_size-1)]);

    //now train the set
    int numcorrect=0, nn_classification;
    double temp_error1, temp_error2, temp_error3, error;
    for(int i=0; i<set_size; i++)
    {
        nn_classification=train(set[i], temp_error1, temp_error2, temp_error3);
        int actual_classification;
        if(isClassA(set[i]))
            actual_classification=0;
        if(isClassB(set[i]))
            actual_classification=1;
        if(isClassC(set[i]))
            actual_classification=2;
        if(nn_classification==actual_classification)
            numcorrect++;
        error1+=temp_error1;
        error2+=temp_error2;
        error3+=temp_error3;
    }
    error=error1+error2+error3;
    return error;
}

void swap(double*& item1, double*& item2)
{
    double temp;
    for(int i=0; i<5; i++)
    {
        temp=item1[i];
        item1[i]=item2[i];
        item2[i]=temp;
    }
}
