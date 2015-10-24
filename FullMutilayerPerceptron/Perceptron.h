#ifndef __FullMutilayerPerceptron__Perceptron__
#define __FullMutilayerPerceptron__Perceptron__

#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
using namespace std;


struct neuron {
	float threshold, output, deltaT, prevDeltaT, resilienceT;
	vector<float> weights, deltaW, prevDeltaW, resilienceW;
    
	neuron() :threshold(((rand() % 100) / 100.0)), resilienceT(1){reset();}
    
	void reset()
	{
		deltaT = 0;
		for (int i = 0; i < deltaW.size(); i++)
			deltaW[i] = 0;
	}
    
	void activationFunction(float input)
	{
		output = tanh(input - threshold);
	}
    
    float getActivationFunctionDerivative()
	{
		return 1 - output*output;
	}
    
    void applyChanges()
    {
        for (int k=0; k<weights.size(); k++)
        {
            if (((signbit(deltaW[k])) == (signbit(prevDeltaW[k]))))
            {
                resilienceW[k] = min(resilienceW[k]*1.2, 50.0);
                weights[k] += resilienceW[k] * deltaW[k];
            }
            else
            {
                weights[k] -= resilienceW[k] * prevDeltaW[k];
                resilienceW[k] = max(resilienceW[k]*0.5, 1.0E-6);
            }
        }
        if (((signbit(deltaT)) == (signbit(prevDeltaT))))
        {
            resilienceT = min(resilienceT*1.2, 50.0);
            threshold += resilienceT * deltaT;
        }
        else
        {
            threshold -= resilienceT * prevDeltaT;
            resilienceT = max(resilienceT*0.5, 1.0E-6);
        }
        deltaW.swap(prevDeltaW);
        swap(deltaT, prevDeltaT);
        
        reset();
    }
    
};

class Perceptron
{
    float learnRate, MSE;

    void forwardPass(const vector<float>& inputSet);
    void backwardPass(const vector<float>& inputSet, const vector<float>& targetSet);
public:
    unsigned int epochCount;
    vector< vector<neuron> > network;

    Perceptron(int inSize, float learningRate, int numberOfLayers, const vector<int>& layerSizes);
    void trainNN(const vector<float>& inputSet, const vector<float>& targetSet);
    float applyEpoch();
    vector<float> getOutput(const vector<float>& inputSet);


};

#endif