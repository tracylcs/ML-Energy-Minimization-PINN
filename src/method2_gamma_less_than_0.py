import tensorflow as tf
from tensorflow.keras import activations
from energy_pinn import *

method = 2

# Fixed parameters
N_train = 128
N_test = 10000
epochs = 100

# Adopted set of parameters
num_layers = 1
layer_size = 1
activation_func = 'relu'
activation_func_name = 'relu'
optimizer = tf.keras.optimizers.legacy.SGD(lr=0.1)
optimizer_name = 'SGD'
learning_rate = 0.1

# Part 0: Evaluate the adopted parameters at gamma = -6
gamma = -6
model, x, x_lb, x_ub, weights = init_model(num_layers, layer_size, N_train, activation_func)
model0, mse0 = fit(model, gamma, method, x, x_lb, x_ub, weights, optimizer, N_test, epochs)

plot([model0], gamma, method, 20000, [mse0], None, None, epochs)

# Part 1: Compare the results for different gammas
gammas = [-30, -14, -6, -2, 0]
models, mses = [], []
for i in range(len(gammas)):
    if (gammas[i] == gamma):
        models.append(model0)
        mses.append(mse0)
        continue
    model, x, x_lb, x_ub, weights = init_model(num_layers, layer_size, N_train, activation_func)
    model, mse = fit(model, gammas[i], method, x, x_lb, x_ub, weights, optimizer, N_test, epochs)
    models.append(model)
    mses.append(mse)

plot(models, gammas, method, 20000, mses, gammas, 'gamma', epochs, variable='gamma')

# Part 2: Parameter testing (taking gamma = -6 as an example)

# (a) Compare the results of using different numbers of hidden layers
nums_layers = [1, 2, 3]
models, mses = [], []
for i in range(len(nums_layers)):
    if (nums_layers[i] == num_layers):
        models.append(model0)
        mses.append(mse0)
        continue
    model, x, x_lb, x_ub, weights = init_model(nums_layers[i], layer_size, N_train, activation_func)
    model, mse = fit(model, gamma, method, x, x_lb, x_ub, weights, optimizer, N_test, epochs)
    models.append(model)
    mses.append(mse)

plot(models, gamma, method, 20000, mses, nums_layers, 'Number of hidden layers', epochs)

# (b) Compare the results of using different sizes of hidden layer
layer_sizes = [1, 2, 3]
models, mses = [], []
for i in range(len(layer_sizes)):
    if (layer_sizes[i] == layer_size):
        models.append(model0)
        mses.append(mse0)
        continue
    model, x, x_lb, x_ub, weights = init_model(num_layers, layer_sizes[i], N_train, activation_func)
    model, mse = fit(model, gamma, method, x, x_lb, x_ub, weights, optimizer, N_test, epochs)
    models.append(model)
    mses.append(mse)

plot(models, gamma, method, 20000, mses, layer_sizes, 'Size of hidden layer', epochs)

# (c) Compare the results of using different activation functions
activation_funcs = ['relu', 'sigmoid', 'tanh', tf.nn.leaky_relu, 'elu']
activation_func_names = ['relu', 'sigmoid', 'tanh', 'leaky relu', 'elu']
models, mses = [], []
for i in range(len(activation_funcs)):
    if (activation_func_names[i] == activation_func_name):
        models.append(model0)
        mses.append(mse0)
        continue
    model, x, x_lb, x_ub, weights = init_model(num_layers, layer_size, N_train, activation_funcs[i])
    model, mse = fit(model, gamma, method, x, x_lb, x_ub, weights, optimizer, N_test, epochs)
    models.append(model)
    mses.append(mse)

plot(models, gamma, method, 20000, mses, activation_func_names, 'Activation function', epochs)

# (d) Compare the results of using different optimizers
optimizers = [tf.keras.optimizers.legacy.SGD(lr=0.1), 
              tf.keras.optimizers.legacy.Adagrad(lr=0.001), 
              tf.keras.optimizers.legacy.Adam(lr=0.001), 
              tf.keras.optimizers.legacy.RMSprop(lr=0.001)]
optimizer_names = ['SGD', 'Adagrad', 'Adam', 'RMSprop']
models, mses = [], []
for i in range(len(optimizers)):
    if (optimizer_names[i] == optimizer_name):
        models.append(model0)
        mses.append(mse0)
        continue
    model, x, x_lb, x_ub, weights = init_model(num_layers, layer_size, N_train, activation_func)
    model, mse = fit(model, gamma, method, x, x_lb, x_ub, weights, optimizers[i], N_test, epochs)
    models.append(model)
    mses.append(mse)

plot(models, gamma, method, 20000, mses, optimizer_names, 'Optimizer', epochs)

# (e) Compare the results of using different learning rates
learning_rates = [0.05, 0.1, 0.2]
optimizers_lr = [tf.keras.optimizers.legacy.SGD(lr=lr) for lr in learning_rates]
models, mses = [], []
for i in range(len(optimizers_lr)):
    if (learning_rates[i] == learning_rate):
        models.append(model0)
        mses.append(mse0)
        continue
    model, x, x_lb, x_ub, weights = init_model(num_layers, layer_size, N_train, activation_func)
    model, mse = fit(model, gamma, method, x, x_lb, x_ub, weights, optimizers_lr[i], N_test, epochs)
    models.append(model)
    mses.append(mse)

plot(models, gamma, method, 20000, mses, learning_rates, 'Learning rate', epochs)
