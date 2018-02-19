# keras-to-cpp
Convert a Keras model to a header file and evaluate in c++

Keras is a great tool to easily build a neural network, but what if you need to deploy in an embedded system? You'd need to export the model, extract the weights, save them into a header file, and then load them into a c/c++ program to do forward propagation. That's a lot of steps...this project automates the entire process.

The code makes some assumptions about the model:
1. It is a binary classification problem
2. There is only a single hidden layer (i.e., a shallow neural network)
3. ReLU activation is used 
4. Probabilities are derived using the sigmoid function 

These assumptions can definitely be made more generalized, but to keep things simple I've stuck with the above assumptions which are relatively reasonable for embedded applications. To make things especially clear, here is the basic breakdown of the neural network architecture:
<img src=https://github.com/nerajbobra/keras-to-cpp/blob/master/images/architecture.jpg alt="Architecture" width="1000">

Here's a breakdown of each script in this project:

#### train_model
This script is mainly here just as an example of training a model. It's a useful reference to make sure you're defining the model with the assumed architecture. This script outputs a model file that's loaded by the next script. StandardScaler is also used to scale the features to unit variance and zero mean, so that is also saved for later use along with the probability threshold that defines which class to assign a given probability.

#### export_model_to_header
This script loads the model file and extracts various weights. It also loads StandardScaler parameters and the probability threshold value. There are four sets of weights: W0, W1, W2, and W3 (see image of architecture for an explanation of each of these weights). The output looks along the lines of the following:
<img src=https://github.com/nerajbobra/keras-to-cpp/blob/master/images/example_header.jpg alt="Example Header" width="1000">

#### main.cpp
This script loads the header file generated by the previous script into an XCode c++ project. The c++ code implements forward propagation as described in the architecture description above. There is a #define variable ENABLE_UNIT_TESTING. If enabled, the c++ file will expect feature inputs from the command line. If commented out, it will compute probabilities directly from within the c++ file. See the c++ file for more detailed comments.

#### test_model
This script takes the model file generated by keras, and runs the features through the built-in functions to get probabilities and corresponding class outputs. It also passes the features to the c++ forward propagation and gets those probabilities and corresponding class outputs. It then overlays the two pairs of outputs onto separate plots to visually confirm the two implementations provide identical results. The following are example outputs:
<img src=https://github.com/nerajbobra/keras-to-cpp/blob/master/images/keras_vs_cpp_probability.jpg alt="Keras vs C++: Probabilities" width="600">
<img src=https://github.com/nerajbobra/keras-to-cpp/blob/master/images/keras_vs_cpp_class.jpg alt="Keras vs C++: Class Outputs" width="600">
