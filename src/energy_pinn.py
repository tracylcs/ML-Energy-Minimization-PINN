import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, initializers

def init_model(num_layers, layer_size, N_train, activation):
    layer_sizes = [1] + [layer_size for i in range(num_layers)] + [1]
    model = neural_net(layer_sizes, activation)
    lb = np.array([0.0])
    ub = np.array([1.0])
    x_lb = tf.convert_to_tensor(lb, dtype=tf.float32)
    x_ub = tf.convert_to_tensor(ub, dtype=tf.float32)
    x = lb + (ub-lb)*np.reshape(np.linspace(0,1,N_train),(N_train,1))
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    weights = tf.Variable([[1.0],[1.0]])
    
    return model, x, x_lb, x_ub, weights

def u_exact(gamma, x):
    u = np.zeros(x.shape)
    if (gamma <= 0 or gamma >= 1):
        for i in range(u.shape[0]):
            u[i,0] = x[i,0] * gamma
    else:
        for i in range(u.shape[0]):
            if (x[i,0] > 1 - gamma):
                u[i,0] = x[i,0] - 1 + gamma
            else:
                u[i,0] = 0
    return u

def neural_net(layer_sizes, activation):
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(layer_sizes[0],)))
    for width in layer_sizes[1:-1]:
        model.add(layers.Dense(width, activation=activation, 
                               kernel_initializer=initializers.glorot_normal(seed=21)))
    model.add(layers.Dense(layer_sizes[-1], activation=None,
                           kernel_initializer=initializers.glorot_normal(seed=21)))
    return model

def fit(model, gamma, method, x, x_lb, x_ub, weights, tf_optimizer, N_test, epochs):
    x_eva = np.linspace(x_lb, x_ub, N_test)
    exact = u_exact(gamma, x_eva)
    mse = np.zeros(epochs+1)
    MSE = tf.keras.losses.MeanSquaredError()
    
    start_time = time.time()
    print("Starting training")

    for epoch in range(epochs):
        
        pred = predict(model, gamma, x_eva)
        u_pred = output_transform(model, gamma, method, x_eva, pred)
        mse[epoch] = MSE(exact, u_pred)
        if (epoch%100 == 0):
            elapsed = time.time() - start_time
            print('Iter: %d, Time: %.2f, mse: %.10f' % (epoch, elapsed, mse[epoch]))
            start_time = time.time()
            
        with tf.GradientTape(persistent=True) as tape:
            loss_value = loss(model, gamma, method, x, x_lb, x_ub, weights)
            grads = tape.gradient(loss_value, model.trainable_variables)
            grads_weights = tape.gradient(loss_value, weights)
            
        tf_optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        if gamma < 0 or gamma > 1:
            tf_optimizer.apply_gradients(zip([-grads_weights], [weights]))
        
        del tape

    pred = predict(model, gamma, x_eva)
    u_pred = output_transform(model, gamma, method, x_eva, pred)
    mse[epochs] = MSE(exact, u_pred)
    elapsed = time.time() - start_time
    print('Iter: %d, Time: %.2f, mse: %.10f' % (epochs, elapsed, mse[epoch]))
    
    return model, mse
    
def predict(model, gamma, x):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    u = model(x)
    return u.numpy()

def output_transform(model, gamma, method, x, pred):
    if (method == 1):          # output transform u(x) = u_bar(x) + gamma*x
        return pred + gamma*x
    elif (method == 2):        # output transform u(x) = u_bar_hat(1-x) + gamma*x
        pred_rev = np.flip(pred)
        return pred_rev + gamma*x
    elif (method == 3):        # no transform required
        return pred 
    elif (method == 4):        # output transform u(x) = v(1-x) + x - 1 + gamma
        pred_rev = np.flip(pred)      
        return pred_rev + x - (1-gamma)*np.ones((x.shape[0],1))
    
def loss(model, gamma, method, x, x_lb, x_ub, weights):
    energy = E_u(model, gamma, method, x)
    pred = model(np.array([x_lb,x_ub]))    
    if (method == 1 or method == 2):
        loss_b = tf.reduce_mean(weights[0]*tf.square(pred[0])+
                                weights[1]*tf.square(pred[1]))  
    elif (method == 3):
        loss_b = tf.reduce_mean(weights[0]*tf.square(pred[0])+
                                weights[1]*tf.square(pred[1]-gamma))
    elif (method == 4):
        loss_b = tf.reduce_mean(weights[0]*tf.square(pred[0])+
                                weights[1]*tf.square(pred[1]-1+gamma))         
    loss_e = tf.reduce_mean(energy)
    return  loss_e + loss_b

def E_u(model, gamma, method, x):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        if (method == 1):
            u_bar = model(x)
        elif (method == 2):
            u_bar = model(1-x)
        elif (method == 3):
            u_bar = model(x) - gamma*x
        elif (method == 4):
            u_bar = model(1-x) - (1-gamma)*(1-x)
        u_bar_x = tape.gradient(u_bar, x)
    ones = tf.convert_to_tensor(np.ones(u_bar_x.shape), dtype=tf.float32)
    w = (u_bar_x+gamma*ones)**2 * ((1-gamma)*ones-u_bar_x)**2
    del tape
    return w

def plot(models, gamma, method, num_pts, mses, labels, legend_title, epochs, variable=''):
    
    fig = plt.figure(figsize=(10,4))
    spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[3, 4])
    ax1 = fig.add_subplot(spec[0])
    ax2 = fig.add_subplot(spec[1])
    
    x_plt = np.linspace(0,1,num_pts)
    x_plt_ = np.reshape(x_plt,(num_pts,1))
    for i in range(len(models)):
        if (variable=='gamma'):
            u = output_transform(models[i], gamma[i], method, x_plt_, predict(models[i], gamma[i], x_plt_))
        else:
            u = output_transform(models[i], gamma, method, x_plt_, predict(models[i], gamma, x_plt_))
        ax1.plot(x_plt, u)
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$u(x)$')
    
    t = epochs
    for i in range(len(mses)):
        if (labels is not None):
            ax2.plot(np.linspace(0,t,t+1), mses[i], label=f"labels[i]")
        else:
            ax2.plot(np.linspace(0,t,t+1), mses[i])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')    
    
    if (legend_title=='gamma'):
        fig.legend(loc='upper right', title=r'$\gamma$')
    elif (legend_title is not None):
        fig.legend(loc='upper right', title=legend_title)
    fig.tight_layout()
    if (legend_title is not None):
        fig.savefig(f"../img/{method}_{legend_title}")
    else:
        fig.savefig(f"../img/{method}")