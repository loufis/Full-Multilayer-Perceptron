#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <opencv/cv.hpp>
#include "Perceptron.h"
using namespace std;

int class1Count = 0, class2Count = 0;
bool usePolar = false;

// ret[i][0] = theta , ret[i][1] = ro
vector< vector<float> > cartesianToPolar(const vector< vector<float> >& input)
{
    vector< vector<float> > ret(input.size());
    for (int i=0; i<input.size(); i++)
    {
        ret[i].resize(2);
        ret[i][0] = atan(input[i][1] / input[i][0]);
        ret[i][1] = sqrt(input[i][0]*input[i][0] + input[i][1]*input[i][1]);
    }
    return ret;
}

void readFile(const string& path, vector< vector<float> >& input, vector< vector<float> >& target)
{
    ifstream in(path.c_str());
    
    if (in.fail())
        cout << "Failed to open file";
    
    string s;
    for (int i=0; i<6; i++)
        getline(in, s);
    
    while (!in.eof())
    {
        vector<float> tempIn(2), tempTarg(1);
        int t;
        in >> tempIn[0] >> tempIn[1] >> t;
        if (t-1)
        {
            class1Count++;
            tempTarg[0] = -1;
        }
        else
        {
            class2Count++;
            tempTarg[0] = 1;
        }
        input.push_back(tempIn);
        target.push_back(tempTarg);
    }
}


int main()
{
    ios_base::sync_with_stdio(0);
    vector< vector<float> > input, target;
    readFile("/Users/Heba/Documents/FullMutilayerPerceptron/FullMutilayerPerceptron/testSpiral.txt", input, target);

    vector < vector <float> > polar = cartesianToPolar(input);
    if (usePolar)
    {
        for (int i=0; i<input.size(); i++)
        {
            input[i].push_back(polar[i][0]);
            input[i].push_back(polar[i][1]);
        }
    }
    
    vector<int> layers(3);
    layers[0] = 10;
    layers[1] = 10;
    layers[2] = 1;
    int inputCount = 2;
    if (usePolar) inputCount = 4;
    Perceptron p(inputCount, 0.0005, 3, layers);
    float err;
    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    unsigned int epochLimit = 1000000;
    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    // Training Loop
    //////////////////////////////////////////////////////////////
    do
    {
        for (int i=0; i<input.size(); i++)
            p.trainNN(input[i], target[i]);
        err = p.applyEpoch();
        //cout << p.epochCount << ' ' << err << endl;
    } while (err > 1 && p.epochCount <= epochLimit);
    //////////////////////////////////////////////////////////////
    
    // Print Epochs and Errors
    //////////////////////////////////////////////////////////////
    int cat1ErrCount = 0, cat2ErrCount = 0;
    for (int i=0; i<input.size(); i++)
    {
        vector<float> outputs = p.getOutput(input[i]);
        
        if ((outputs[0] > 0) != (target[i][0] > 0))
        {
            if (outputs[0] > 0)
                cat1ErrCount++;
            else
                cat2ErrCount++;
        }
        
    }
    cout << "Epochs: " << p.epochCount;
    cout << endl << endl << "Final Mean Square Error: " << err;
    cout << endl << endl << "Number of Wrong Category 1 Classifications: " << cat1ErrCount << " / " << class1Count;
    cout << endl << endl << "Number of Wrong Category 2 Classifications: " << cat2ErrCount << " / " << class2Count << endl << endl;
    //////////////////////////////////////////////////////////////
        
    // Print Weights
    //////////////////////////////////////////////////////////////
    cout << endl << endl << "Final Weights and Bias: " << endl << endl;
    for (int i=0; i<p.network.size(); i++)
    {
        for (int j=0; j<p.network[i].size(); j++)
        {
            for (int k=0; k<p.network[i][j].weights.size(); k++)
                cout << p.network[i][j].weights[k] << ' ';
            cout << p.network[i][j].threshold << endl;
        }
        cout << endl;
    }
    //////////////////////////////////////////////////////////////
    

    
    
    //@@@@@@@@@@@@@@@@@@@@@@@
    int sizeOfImage = 1000;
    //@@@@@@@@@@@@@@@@@@@@@@@

    cv::Mat1f boundaryLayer = cv::Mat1f::zeros(sizeOfImage,sizeOfImage);

    /*
    cout << endl;
    for (int i=0; i<input.size(); i++)
    {
        vector<float> outputs = p.getOutput(input[i]);
        double norm = sqrt(input[i][0]*input[i][0] + input[i][1]*input[i][1]);
        cout << norm << ' ' << input[i][0] << ' ' << input[i][0]/norm << endl;
        if (outputs[0] < outputs[1])
        {
            cout << int(input[i][0]/norm)*sizeOfImage << ' ' << int(input[i][1]/norm)*sizeOfImage << endl;
            boundaryLayer[int(((input[i][0]/norm) + 2)* (sizeOfImage/4))][int(((input[i][1]/norm) + 2)*(sizeOfImage/4))] = 255;
        }
        cout << endl;
    }
    */
    vector < vector <float> > coord(sizeOfImage*sizeOfImage);
    for (int y=0; y<sizeOfImage; y++)
    {
        for (int x=0; x<sizeOfImage; x++)
        {
            coord[y*sizeOfImage+x].push_back((x*1.0)/(sizeOfImage/4) - 2);
            coord[y*sizeOfImage+x].push_back((y*1.0)/(sizeOfImage/4) - 2);
        }
    }
    if (usePolar)
    {
        polar = cartesianToPolar(coord);
        
        for (int i=0; i<coord.size(); i++)
        {
            coord[i].push_back(polar[i][0]);
            coord[i].push_back(polar[i][1]);

        }
    }

    for (int y=0; y<sizeOfImage; y++)
    {
        for (int x=0; x<sizeOfImage; x++)
        {
            vector<float> output = p.getOutput(coord[y*sizeOfImage + x]);
            boundaryLayer[y][x] = (output[0] < 0) * 255;
        }
    }
    
    
    cv::namedWindow("Display window", cv::WINDOW_NORMAL);
    cv::imshow("Display window", boundaryLayer);
    cv::waitKey(0);
    
        
        
    
    return 0;
}

