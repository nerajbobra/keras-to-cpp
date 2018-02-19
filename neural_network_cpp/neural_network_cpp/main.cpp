
#include <iostream>
#include <vector>
#include <math.h>
#include "model_params.h"

//to test with python, leave ENABLE_UNIT_TESTING uncommented
//otherwise, comment ENABLE_UNIT_TESTING to evaluate the model directly from this project
#define ENABLE_UNIT_TESTING

//define a struct that will hold the output results from the neural network
struct NeuralNetworkResult {
    float probability;
    int predicted_class;
};
NeuralNetworkResult output;

//define an array to hold intermediate outputs from the neural network
float out[NUM_HIDDEN_NEURONS];

//function declarations
void calclate_probability(float *X);

//the main function is broken into two parts, one for unit testing and the other for standard usage
int main(int argc, const char * argv[]) {
#ifndef ENABLE_UNIT_TESTING
    //define a random set of features and evaluate it
    float X[NUM_FEATS] = {1.79900000e+01,   1.03800000e+01,   1.22800000e+02,
                            1.00100000e+03,   1.18400000e-01,   2.77600000e-01,
                            3.00100000e-01,   1.47100000e-01,   2.41900000e-01,
                            7.87100000e-02,   1.09500000e+00,   9.05300000e-01,
                            8.58900000e+00,   1.53400000e+02,   6.39900000e-03,
                            4.90400000e-02,   5.37300000e-02,   1.58700000e-02,
                            3.00300000e-02,   6.19300000e-03,   2.53800000e+01,
                            1.73300000e+01,   1.84600000e+02,   2.01900000e+03,
                            1.62200000e-01,   6.65600000e-01,   7.11900000e-01,
                            2.65400000e-01,   4.60100000e-01,   1.18900000e-01};
    calclate_probability(X);
    printf("calculated probability: %f, predicted class: %i\n", output.probability, output.predicted_class);
#else
    //copy the data from the command line into an array to be passed to the calculate_probability function
    float X[NUM_FEATS];
    for (int i=0; i<NUM_FEATS; i++) {
        X[i] = atof(argv[i+1]);
    }
    
    calclate_probability(X);
    printf("%f, %i\n", output.probability, output.predicted_class);
#endif
}

//apply the neural network and return the probability
void calclate_probability(float *X) {
    //subtract by mu, divide by sigma
    for (int i=0; i<NUM_FEATS; i++) {
        X[i] = (X[i]-mu[i])/sigma[i];
    }
    
    //clear the temp array
    memset(out, 0, sizeof(out)*NUM_FEATS);
    
    //do multiplication with W0 and add W1
    //also do a relu activation
    float temp;
    for (int i=0; i<NUM_HIDDEN_NEURONS; i++) {
        temp = 0;
        for (int j = 0; j<NUM_FEATS; j++) {
            temp += X[j]*W0[j][i];
        }
        out[i] = temp + W1[i];
        
        //do relu
        out[i] = out[i] > 0 ? out[i] : 0;
    }
    
    //do multiplication with W2
    float prob = 0;
    for (int i=0; i<NUM_HIDDEN_NEURONS; i++) {
        prob += out[i]*W2[i];
    }
    
    //add W3
    prob += W3;
    
    //do the softmax
    prob = exp(prob)/(exp(prob) + 1);
    
    //get the class prediction
    int predicted_class = prob > thresh ? 1 : 0;
    
    //package the output
    output.probability = prob;
    output.predicted_class = predicted_class;
}



