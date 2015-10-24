#include "Perceptron.h"


Perceptron::Perceptron(int inSize, float learningRate, int numberOfLayers, const vector<int>& layerSizes) :learnRate(learningRate), epochCount(0), MSE(0)
{
	srand(time(NULL));
	network.resize(numberOfLayers);
	for (int i = 0; i < numberOfLayers; i++)
		network[i].resize(layerSizes[i]);
    
    for (int j=0; j<network[0].size(); j++)
    {
        network[0][j].weights.resize(inSize);
        network[0][j].deltaW.resize(inSize);
        network[0][j].prevDeltaW.resize(inSize);
        network[0][j].resilienceW.resize(inSize, 1);
        for (int k=0; k<network[0][j].weights.size(); k++)
            network[0][j].weights[k] = ((rand() % 100) / 100.0);
    }
    
    for (int i = 1; i < numberOfLayers; i++)
        for (int j=0; j<network[i].size(); j++)
        {
            network[i][j].weights.resize(layerSizes[i-1]);
            network[i][j].deltaW.resize(layerSizes[i-1]);
            network[i][j].prevDeltaW.resize(layerSizes[i-1]);
            network[i][j].resilienceW.resize(layerSizes[i-1], 1);
            for (int k=0; k<network[i][j].weights.size(); k++)
                network[i][j].weights[k] = ((rand() % 100) / 100.0);
        }
}



void Perceptron::forwardPass(const vector<float>& inputSet)
{
    for (int j=0; j<network[0].size(); j++)
    {
        float sum = 0;
        for (int k=0; k<inputSet.size(); k++)
        {
            sum += inputSet[k] * network[0][j].weights[k];
        }
        network[0][j].activationFunction(sum);
    }
    
    
    for (int i=1; i<network.size(); i++)
    {
        for (int j=0; j<network[i].size(); j++)
        {
            float sum = 0;
            for (int k=0; k<network[i-1].size(); k++)
            {
                sum += network[i-1][k].output * network[i][j].weights[k];
            }
            network[i][j].activationFunction(sum);
        }
    }
}



void Perceptron::backwardPass(const vector<float>& inputSet, const vector<float>& targetSet)
{
    vector <float> currentLocalGradients(network[network.size()-1].size());

    if (network.size() > 1)
    {
        for (int j=0; j<network[network.size()-1].size(); j++)
        {
            currentLocalGradients[j] = network[network.size()-1][j].getActivationFunctionDerivative()
                                        *(targetSet[j] - network[network.size()-1][j].output);

            for (int k=0; k<network[network.size()-1][j].deltaW.size(); k++)
            {
                network[network.size()-1][j].deltaW[k] += learnRate * network[network.size()-2][k].output * currentLocalGradients[j];
            }
            network[network.size()-1][j].deltaT -= learnRate * currentLocalGradients[j];
        }
    

        for (int i=network.size()-2; i>0; i--)
        {
            vector<float> localGradientsOfNextLayer;
            localGradientsOfNextLayer.swap(currentLocalGradients);
            currentLocalGradients.resize(network[i].size());
            
            for (int j=0; j<network[i].size(); j++)
            {
                currentLocalGradients[j] = 0;
                for (int n=0; n<network[i+1].size(); n++)
                {
                    currentLocalGradients[j] += network[i+1][n].weights[j] * localGradientsOfNextLayer[n];
                }
                currentLocalGradients[j] *= network[i][j].getActivationFunctionDerivative();
                
                for (int k=0; k<network[i][j].deltaW.size(); k++)
                {
                    network[i][j].deltaW[k] += learnRate * network[i-1][k].output * currentLocalGradients[j];
                }
                network[i][j].deltaT -= learnRate * currentLocalGradients[j];
            }
        }
        
        vector<float> localGradientsOfNextLayer;
        localGradientsOfNextLayer.swap(currentLocalGradients);
        currentLocalGradients.resize(network[0].size());
        for (int j=0; j<network[0].size(); j++)
        {
            currentLocalGradients[j] = 0;
            for (int n=0; n<network[1].size(); n++)
            {
                currentLocalGradients[j] += network[1][n].weights[j] * localGradientsOfNextLayer[n];
            }
            currentLocalGradients[j] *= network[0][j].getActivationFunctionDerivative();
            
            for (int k=0; k<inputSet.size(); k++)
            {
                network[0][j].deltaW[k] += learnRate * inputSet[k] * currentLocalGradients[j];
            }
            network[0][j].deltaT -= learnRate * currentLocalGradients[j];
        }
    }
    else
    {
        for (int j=0; j<network[0].size(); j++)
        {
            currentLocalGradients[j] = network[0][j].getActivationFunctionDerivative()
                                        *(targetSet[j] - network[0][j].output);
            
            for (int k=0; k<network[0][j].deltaW.size(); k++)
            {
                network[0][j].deltaW[k] += learnRate * inputSet[k] * currentLocalGradients[j];
            }
            network[0][j].deltaT -= learnRate * currentLocalGradients[j];
        }
    }
    
}



void Perceptron::trainNN(const vector<float>& inputSet, const vector<float>& targetSet)
{
    forwardPass(inputSet);

    float curMSE = 0;
    for (int j=0; j<network[network.size()-1].size(); j++)
    {
        float error = (targetSet[j]-network[network.size()-1][j].output);
        curMSE += error * error;
    }
    MSE += curMSE/2;
    
    backwardPass(inputSet, targetSet);
}

vector<float> Perceptron::getOutput(const vector<float>& inputSet)
{
    forwardPass(inputSet);
    vector<float> ret(network[network.size()-1].size());
    for (int i=0; i<ret.size(); i++)
        ret[i] = network[network.size()-1][i].output;
    return ret;
}


float Perceptron::applyEpoch()
{
    
    for (int i=0; i<network.size(); i++)
    {
        for (int j=0; j<network[i].size(); j++)
        {
            network[i][j].applyChanges();

        }
    }
    
    epochCount++;
    float curMSE = MSE;
    MSE = 0;
    return curMSE;
}










