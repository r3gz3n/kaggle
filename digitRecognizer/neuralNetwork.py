import pandas as pd
import numpy as np
from scipy.optimize import fmin_bfgs


def sigmoid(z):
    """
    Computes sigmoid of z
    Assumes z to be a numpy array
    """
    g = 1 / (1 + np.exp(-z))
    return g


def sigmoidGradient(z):
    """
    Returns gradient of sigmoid function
    Assumes z to be a numpy array
    """
    g = sigmoid(z) * (1 - sigmoid(z))
    return g


def randInitializeWeights(L_in, L_out):
    """
    Randomly initializes weights of a layer with L_in incoming connections and L_out outgoing connections
    """
    epsilon_init = 0.12
    W = np.random.rand(L_in, L_out + 1) * 2 * epsilon_init - epsilon_init
    return W


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, l):
    """
    Computes the cost and gradient of the neural network
    """
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)

    m = X.shape[0]

    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    a1 = np.hstack([np.ones((m, 1)), X])
    z2 = a1.dot(Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack([np.ones((m, 1)), a2])
    z3 = a2.dot(Theta2.T)
    h = sigmoid(z3)

    y1 = np.array([[]]*m)
    for i in range(num_labels):
      condlist = [y == i]
      choicelist = [1]
      y1 = np.hstack([y1, np.select(condlist, choicelist).reshape((m, 1))])

    for i in range(num_labels):
      J += ((-y1[:, i]).T.dot(np.log(h[:, i])) - (1 - y1[:, i]).T.dot(np.log(1 - h[:, i]))) / m

    J += (l * np.sum((Theta1[:, 1:] ** 2)[:])) / (2 * m)
    J += (l * np.sum((Theta2[:, 1:] ** 2)[:])) / (2 * m)

    delta1 = np.zeros(Theta1.shape)
    delta2 = np.zeros(Theta2.shape)
    for t in range(m):
        a1 = np.hstack([np.ones((1, 1))[0], X[t]]).reshape(1, input_layer_size + 1)
        z2 = a1.dot(Theta1.T)
        a2 = sigmoid(z2)
        a2 = np.hstack([np.ones((1, 1)), a2])
        z3 = a2.dot(Theta2.T)
        h = sigmoid(z3)
        error3 = (h - y1[t,:])
        error2 = np.matmul(error3, Theta2)[:, 1:] * sigmoidGradient(z2)
        delta1 = delta1 + np.matmul(error2.T, a1)
        delta2 = delta2 + np.matmul(error3.T, a2)

    Theta1_grad = delta1 / m
    Theta1_grad[:, 1:] += (l * Theta1[:, 1:]) / m
    Theta2_grad = delta2 / m
    Theta2_grad[:, 1:] += (l * Theta2[:, 1:]) / m

    grad = np.hstack([Theta1_grad.flatten(), Theta2_grad.flatten()])

    return J, grad


def computeNumericalGradient(J, theta):
    """
    Computes the numerical gradient of the function J around theta
    """
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for p in range(theta.size):
        # Set perturbation vector
        print(p, theta.size)
        perturb[p] = e
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0
    return numgrad

def checkNNGradients(costFuncGrad, costFunc, l, Theta1, Theta2):
    """
    Creates a small neural network to check the backpropagation gradients
    """
    nn_params = np.hstack([Theta1.flatten(), Theta2.flatten()])
    grad = costFuncGrad(nn_params)
    print(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)

    print([numgrad, grad])
    diff = np.linalg.norm(numgrad-grad) / np.linalg.norm(numgrad+grad)
    print(diff)

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    p = np.zeros((m, 1))

    h1 = sigmoid(np.matmul(np.hstack([np.ones((m, 1)), X]), Theta1.T))
    h2 = sigmoid(np.matmul(np.hstack([np.ones((m, 1)), h1]), Theta2.T))
    p = np.argmax(h2, axis = 1) + 1

    return p

def neuralNetwork(X, y, initial_Theta1, initial_Theta2):

    # Setup the parameters
    input_layer_size = 784
    hidden_layer_size = 35
    num_labels = 10


    initial_nn_params = np.hstack([initial_Theta1.flatten(), initial_Theta2.flatten()])

    l = 1
    def costFunc(theta):
        """
        Short hand for Cost Function
        """
        J, grad = nnCostFunction(theta, input_layer_size, hidden_layer_size, num_labels, X, y, l)
        #print(theta, J, grad)
        return J

    def costFuncGrad(theta):
        """
        Short hand for Cost Function
        """
        J, grad = nnCostFunction(theta, input_layer_size, hidden_layer_size, num_labels, X, y, l)
        return grad
    #print('Checking Gradient ...')
    #checkNNGradients(costFuncGrad, costFunc, l, initial_Theta1, initial_Theta2)


    print("Running Gradient Descent ...")
    nn_params = fmin_bfgs(costFunc, initial_nn_params, maxiter = 10, fprime = costFuncGrad)

    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)
    return Theta1, Theta2

if __name__ == '__main__':

    input_layer_size = 784
    hidden_layer_size = 35
    num_labels = 10

    print('Loading and Visualizing Data ...')

    train = pd.read_csv("train_testdata.csv")

    print('Extracting Features and Label ...')

    initial_y = np.array(train["label"], dtype = pd.Series)
    initial_X = np.array(train[train.columns.difference(['label'])])

    print('Initializing Theta with Random Values ...')
    Theta1 = randInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = randInitializeWeights(num_labels, hidden_layer_size)
    Theta1, Theta2 = neuralNetwork(initial_X, initial_y, Theta1, Theta2)
    """
    for i in range(5):
        p = 2 * i
        X = initial_X[p : p + 2, :]
        y = initial_y[p : p + 2]
        print(i, p)
        Theta1, Theta2 = neuralNetwork(X, y, Theta1, Theta2)
    """
    print("Predicting values ...")
    pred = predict(Theta1, Theta2, initial_X)
    pred = pd.DataFrame(pred, index=range(1, pred.shape[0]+1))
    pred.to_csv("my_solution1.csv")
    print((pred == y).mean() * 100)
