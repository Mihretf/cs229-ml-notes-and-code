
# Single Layer Neural Network
Consists of a raw data, a middle hidden layer that transforms the data and a final output layer that makes the prediction. It forms the basic building block of deep learning. 

# One hot representation
this is a way to translate categories like digits 0 through 9 into numbers that a computer can process without assuming an aritifical ordering. So 3 would be written as, 0,0,0,1,0,0,0,0,0,0 -> It uses one at the correct index of the correct class and places zero on the other parts. 

# Hidden Layer
these are the intermediate neurons residing between the input and output layers. Rather than forcing a human to manually engineer rules or intermediate features, like deciding where a house has a good family size or like tha, hidden layers automatically discover and compute abstract pattersn from the input data. 

# Sigmoid Activation Function
A mathematical squashing function that takes any raw number and compresses it intoa smooth probability range between 0 and 1. it introduces non-linearity, allowing the network to capture complex, curved relationships rather than just drawing straight lines. 

# Softmax function
this is used at the final output layer. It takes a vector of raw output scores and converts them into a clean probability distribution across all K classes, where every score is positive and the total sum equals 1. 

# Cross Entropy loss
A performance metric that measures how badly the network's predicted probability distribution matches the true one hot target label. If the model assigns high probabilty to the correct class, the penality is low, if it guess wrongly with high confidence, the loss sky-rockets. 

# L2  regualarization
A safeguard against overfitting, it adds a penalty to the loss function proportional to the squared magnitude of the models weight. This discourages the network from relying too heavily on any single feature or growing overly complex weights, forcing it to find smoother, more generalizable patterns. 

# Mini-Batch Gradient Descent
an optimization technique where the model updates its weights using a small subset of the training data a mini-batch of size B, at each steo, balancing the speed of SGD and the stability of full-batch gradient descent. 

# How It All Interconnects to Build a Neural Network

The Flow: Raw input data (like a flattened $784$-pixel image) enters the network.  

First Transformation: The inputs are multiplied by the first weight matrix ($W^{[1]}$), adjusted by biases ($b^{[1]}$), and passed through the sigmoid activation function inside the hidden units. This step turns the raw pixels into a rich set of intermediate, non-linear representations.

Second Transformation: Those hidden features are multiplied by the second weight matrix ($W^{[2]}$) to produce raw output scores for every possible digit category.

Probability & Loss: The softmax function converts those raw scores into final class probabilities, and cross-entropy loss evaluates how far off those probabilities were from the true one-hot representation.  

Optimization & Control: During backpropagation, gradients flow backward to update the weights, while L2 regularization keeps those weights grounded so the network generalizes cleanly to unseen data rather than memorizing noise.

def sigmoid(x):
max_val = np.max(x, axis=-1, keepdims=True)
-> This line scans across the columns of our data matrix to find th ehightest number, while keeping the vertical column shape intact so it can be safely subtracted later. 
? Why do we subtract it later, and why is the max value subtracted
    exp_x = np.exp(x - max_val)
    -> this takes our raw input numbers, subtracts the max value we just found from each of them to keep the numbers small and sage and then applies exponential mathematical function, to every item. e to the power of that number
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
-> this calculates the final softmax probabilty distribution. It takes the exponential values, adds up all the values across each row and divides each indivdual value by that row's total sum so that everything proportion out cleanly into probabilites adding up to 1 

def get_initial_params(input_size, num_hidden, num_output):

params = {} -> this creates an empty python dictionary container that will hold all of our neural network's ajustable memory variables like the weight and biases. 
    params['W1'] = np.random.randn(input_size, num_hidden)
-> this creates the first hiden layer's weight matrix, It fills a grid of size input size rows by num_hidden columns with random numbers drawn from a standard bell curve distribution
    params['b1'] = np.zeros((1, num_hidden))
-> this creates the first layers bias vecor as a row of zeros, giving our hidden neurons a neural starting threshold
    params['W2'] = np.random.randn(num_hidden, num_output)
-> this creates the second layers weight matrix, connecting the 300 hidden neurons to the final 10 ouput class categories using random starting weights. 
    params['b2'] = np.zeros((1, num_output))
-> this creates a row of zeros for the final output layer's biases. 
    return params
-> this hands back the completed dictionary packed with all four initialized parameters so the rest of the program canuse them. 


def forward_prop(data, labels, params):

W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

-> these  lines extract the hidden layer's and the output layer's weight and biases fromthe parameter dictionary

    z1 = np.dot(data, W1) + b1
-> this multiplies the input images data by the first weight matrix and adds the hidden bias vector, producing the raw pre-activation scores for the hidden layer. 
    a1 = sigmoid(z1)
-> this passes the raw hidden scores through the activation function, squashing every number into a smooth probabilty range between 0 and 1. 
    z2 = np.dot(a1, W2) + b2
-> this multiplies the hiden layer's activated outputs by the second weight matrix and ads the output bias, produicng raw score predicitions for eah of the 10 digit classes
    output = softmax(z2)
-> this passes the final raw scores throught the softmax function to turm them into valid percentage probabalies for every digit class

    num_examples = data.shape[0]
-> this checks how many image rows are currently inside your data batch and saves that count. 
    # Cross-entropy loss averaged over examples (adding a tiny epsilon to prevent log(0))
    loss = -np.sum(labels * np.log(output + 1e-15)) / num_examples
-> this computes the cross entropy loss error score. It multiplies true labels by the log of the network's predictions, adds a tiny number to prevent computer crashes from taking log of 0, sums everything up, and divides bu the total number of examples to get the average error. 
    return a1, output, loss
-> this sends back the hidden layer activations, the final softmax prediction probablities and the calculated average loss score. 

def backward_prop(data, labels, params, forward_prop_func):
a1, output, loss = forward_prop_func(data, labels, params)
-> this runs forward propagation first so we can obtain the network's predictions and hidden activations needed to calculate our errors. 
    num_examples = data.shape[0]
-> this records the total number of data examples in the current batch. 

    # Delta for output layer (from part a derivation: y_hat - y)

    delta2 = output - labels  # shape: (B, K)
-> this calculates the error gradient for the output layer by simply subtracting the true one-hot labels from the model's softmax predictions. 
    # Gradients for W2 and b2
    grad_W2 = np.dot(a1.T, delta2) / num_examples
-> this computes the gradient for the second weight matrix by multiplying the transposes hidden activations by the output error delta, average across the batch
    grad_b2 = np.sum(delta2, axis=0, keepdims=True) / num_examples
-> this computes the gradient for the output bias vector by summing up the output errors down each column, averaged across the batch. 

    # Delta for hidden layer using backpropagation rule through sigmoid activation
    delta1 = np.dot(delta2, params['W2'].T) * a1 * (1 - a1)
-> this tracks the error backward into the hidden layer, It multiplies output errors back through the second weights, and scales them by the derivative of the sigmoid function

    # Gradients for W1 and b1
    grad_W1 = np.dot(data.T, delta1) / num_examples
-> computing the gradient for the first weight bu multipluing the transposed input data by the hidden layer's error delta, averaged across the batch.
    grad_b1 = np.sum(delta1, axis=0, keepdims=True) / num_examples
->This computes the gradient for the hidden bias vector by summing up the hidden error values down each column, averaged across the batch.

    return {
        'W1': grad_W1,
        'b1': grad_b1,
        'W2': grad_W2,
        'b2': grad_b2
    }

def backward_prop_regularized(data, labels, params, forward_prop_func, reg):

grads = backward_prop(data, labels, params, forward_prop_func)
->This calls the standard backward propagation function first to get all the basic unregularized gradients.
    
    # Add L2 regularization penalty to the weight gradients (biases are excluded)
    grads['W1'] += 2 * reg * params['W1']
-> This adds the $L_2$ regularization penalty directly onto the hidden weight gradients, proportional to the size of the weights and the regularization strength ($\lambda$).
    grads['W2'] += 2 * reg * params['W2']
-> This adds the same $L_2$ regularization penalty onto the output weight gradients.
    return grads
-> This returns the newly updated dictionary containing the regularized gradients (while leaving the bias terms safely unregularized).


def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func):
num_examples = train_data.shape[0]
-> This counts the total number of rows (training images) available in the dataset.
    num_batches = num_examples // batch_size
-> This calculates how many total mini-batches we will process during one complete pass through the data.

    for b in range(num_batches):
-> This starts a loop that will process each mini-batch sequentially one by one.
        start_idx = b * batch_size
-> This determines the starting row index for the current mini-batch chunk.
        end_idx = start_idx + batch_size
-> This determines the ending row index for the current mini-batch chunk.

        x_batch = train_data[start_idx:end_idx]
-> This slices out a specific mini-batch chunk of image data from the full training dataset.
        y_batch = train_labels[start_idx:end_idx]
-> This slices out the matching mini-batch chunk of target labels.
        grads = backward_prop_func(x_batch, y_batch, params, forward_prop_func)
-> This calls our backpropagation function to calculate the gradients for this specific mini-batch chunk.
        # Update parameters using mini-batch gradient descent update rule
        params['W1'] -= learning_rate * grads['W1']
-> This updates the hidden weights by stepping them in the opposite direction of their gradient, scaled down by the learning rate.
        params['b1'] -= learning_rate * grads['b1']
-> This updates the hidden bias vector using the gradient descent rule.
        params['W2'] -= learning_rate * grads['W2']
-> This updates the output weights using the gradient descent rule.
        params['b2'] -= learning_rate * grads['b2']
-> This updates the output bias vector using the gradient descent rule.


# ? why is that we are calculating the gradient for everything 
> In machine learning, gradients are just compass directions telling us which way is "up" (toward higher error) and which way is "down" (toward lower error).Every single parameter in our network—every weight matrix ($W^{[1]}, W^{[2]}$) and every bias vector ($b^{[1]}, b^{[2]}$)—is responsible for making guesses about the handwritten digits. When the network makes a mistake, we need to know how much blame to assign to each individual parameter.If we didn't calculate gradients for everything, we wouldn't know which weights caused the wrong prediction. Calculating the gradient for every parameter allows us to nudge every single one of them slightly in the "correct" direction (downhill toward lower loss) so that next time, the network makes a better prediction
# What do regularized gradients mean
> A regularized gradient is simply a standard gradient that has been punished for letting the weights grow too large.The Problem: Without regularization, a neural network can become overly confident and "memorize" the training data by creating massive weight values that react too sharply to tiny details (overfitting).The Solution ($L_2$ Regularization): When we add regularization to the gradient ($+ 2 \cdot \text{reg} \cdot \text{params}$), we force the gradient to pull the weights back toward zero.The Meaning: It tells the network: "Yes, update this weight to reduce the classification error, but also shrink yourself down a little bit so you don't become too complex or rigid." This keeps the network's decision boundaries smooth, helping it generalize cleanly to completely unseen test images



# The Big Picture: What Training a Neural Network Means

tepping back from individual lines of code, here is the complete macro-level story of what we accomplished across all these functions:

Random Initialization (get_initial_params):
We started by giving the network a blank slate. We filled its weight matrices with random values so that different neurons start off looking for different patterns.

The Prediction Cycle / Forward Propagation (forward_prop):
We took raw pixel data from MNIST images, multiplied them through our weights, passed them through the sigmoid "squashing" gate, and converted final scores into probabilities via softmax. This is the network thinking or making a guess.

Error Measurement:
We compared the network's guess against the true one-hot label using cross-entropy loss to get a single numerical score representing how wrong the model was.

Blame Assignment / Backpropagation (backward_prop):
Using calculus and the chain rule, we walked backward from the output layer to the input layer. This traced how every single weight contributed to the final error, outputting a precise gradient map of corrections.

Updating Memory / Gradient Descent (gradient_descent_epoch):
We took those gradients and adjusted every weight and bias using the learning rate. We repeated this process thousands of times in mini-batches across multiple epochs.

What Makes This Training a Neural Network?
Several key structural elements make this a true neural network rather than a standard linear classifier:

The Hidden Layer (num_hidden = 300): Instead of connecting raw input pixels directly to the output classes (which only allows for straight-line decision boundaries), we route data through an intermediate layer of 300 hidden neurons. These neurons act as feature detectors, learning abstract shapes like loops, lines, and curves.

Non-Linear Activation Functions (sigmoid): If we only multiplied matrices together without a non-linear activation like sigmoid, stacking multiple layers would mathematically collapse into a single boring linear equation. The sigmoid function allows the network to learn complex, curved, non-linear relationships.

Multi-Tiered Optimization: It utilizes backpropagation to automatically propagate error signals across multiple sequential layers of weights simultaneously, allowing hierarchical learning where early layers find edges and later layers combine them into digits.