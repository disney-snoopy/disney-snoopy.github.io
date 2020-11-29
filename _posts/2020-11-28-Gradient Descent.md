# Batch Gradient Descent
Parameter update is based on the cost derivative calculation for entire training data.
## Cost Function
$$
J(\theta) = \frac{1}{2m}\sum_{i=1}^{m} \bigl(h_\theta(x) - y_i \bigr)^2
$$
where $m$ is sample size, $y_i$ is $i^{th}$ sample label, $h_\theta(x)$ is the hypothesis function of x parameterised by $\theta$.\
One of the assumptions of generalised linear model states:
$$h_\theta(x) = E(y|x)$$
where $x$ is sample feature.

## Parameter Update
Parameter update follows maximum likelihood estimation. 
$$
\theta := \theta - \alpha \cdot \frac{\delta J}{\delta\theta}
$$
where $\alpha$ is learning rate.

$$
\frac{\delta J}{\delta\theta} = \frac{1}{m}\sum_{i=1}^{m} \bigl(h_\theta(x)) - y_i \bigr)x_i
$$

Most primitive gradient descent algorithm.

### Advantages
1) Guranteed to choose the direction that has the steepest gradient down to the local or global minima.\
2) Tha algorithm is guranteed to converge.\

### Limitations
1) For one parameter update, the algorithm needs to go through the entire training dataset which can be computationally expensive, especially when the training dataset is large.\
2) Constant learning rate $\alpha$ can lead to painfully slow convergence that leads to longer learning time.

## Example
#### Dataset


```python
pip install nbimporter
```

    Requirement already satisfied: nbimporter in c:\users\bjk\anaconda3\lib\site-packages (0.3.3)
    Note: you may need to restart the kernel to use updated packages.
    


```python
import nbimporter
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange
from design_matrix import *
from normal_eq import normal
```

    Importing Jupyter notebook from normal_eq.ipynb
    


```python
def normal_eq(training_x, training_y):
    dm = design_matrix(training_x)
    y = training_x.reshape(len(training_y), 1)   
    optimum = np.linalg.inv(dm.transpose().dot(dm)).dot(dm.transpose()).dot(training_y)
    return optimum
```


```python
def cost_function(dm, training_y, param):
    outcome = np.dot(dm, param)
    error = outcome - training_y
    cost_history = (error**2).sum()/len(error)
    return cost_history
```


```python
def batch_gd(training_x, training_y, learning_rate, iteration):
    dm = design_matrix(training_x)
    num_param = dm.shape[1]
    param_hist = np.zeros(num_param).reshape(1, num_param)
    outcome = np.dot(dm, param_hist[-1,:].transpose())
    error = outcome - training_y
    cost_hist = (error**2).sum()/len(training_y)
    derivative = ((error.reshape(len(error), 1) * dm).sum(axis = 0))/len(training_y)
    derivative = derivative.reshape(1, num_param)
    for i in range(iteration):
    #for i in tqdm(range(iteration)):
        temp_param = param_hist[-1,:]
        new_param = temp_param - (learning_rate * derivative)
        param_hist = np.vstack((param_hist, new_param))
        outcome = np.dot(dm, param_hist[-1,:].transpose())
        error = outcome - training_y
        new_cost = (error**2).sum()/len(training_y)
        cost_hist = np.append(cost_hist, new_cost)
        derivative = ((error.reshape(len(error), 1) * dm).sum(axis = 0)/len(training_y))
    return cost_hist, param_hist
```


```python
def algo_summary(algo,training_x, training_y, iteration):
    fig, ax = plt.subplots(2,2, figsize = (15,15))
    pred_x = np.arange(min(training_x), max(training_x), 0.01)
    pred_dm = design_matrix(pred_x)
    
    #Subplot (0,0)
    #Prediction vs iteration.
    ax[0,0].scatter(training_x, training_y)
    for iteration in [1, 3, 5, 100, 1000]:
        if algo == 'batch':
            learning_rate = 0.01
            cost_hist, param_hist = batch_gd(training_x, training_y, learning_rate, iteration)
        #if algo == 'stochastic'...
        param = param_hist[-1,:]
        pred_y = np.dot(pred_dm, param)
        ax[0,0].plot(pred_x, pred_y, label = 'Iteration: %d' %iteration)
        ax[0,0].set_title('Prediction vs. Iteration (learning rate = 0.01)',fontsize = 20)
        ax[0,0].set_xlabel('x',fontsize = 20)
        ax[0,0].set_ylabel('y',fontsize = 20)
    ax[0,0].legend()
    
    #Subplot(0, 1)
    #Cost vs learning rate.
    
    for learning_rate in [0.001, 0.005, 0.05]:
        iteration = 80
        if algo == 'batch':
            cost_hist, param_hist = batch_gd(training_x, training_y, learning_rate, iteration)
        #----
        ax[0,1].plot(cost_hist,'--', label = 'learning rate: %.3f' %learning_rate, alpha = 0.5)
        ax[0,1].set_title('Cost vs Learning rate',fontsize = 20)
        ax[0,1].set_ylabel('Cost',fontsize = 20)
        ax[0,1].set_xlabel('Iteration',fontsize = 20)
    ax[0,1].legend()
    
    #Subplot (1,0)
    #Mesh grid of cost.
    
        #Finding the optimal paramters using normal equation.
    optimum = normal_eq(training_x, training_y)

        #Creating a meshgrid.
    xmin, xmax, xstep = -5, 15, .1
    ymin, ymax, ystep = -5, 15, .1
    intersect, gradient = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
    
        #Calculating cost for each point on meshgrid.
    cost_list = []
    for i in range(len(intersect)):
        for j in range(len(gradient)):
            param = (intersect[i,j], gradient[i,j])
            cost_list.append(cost_function(dm,training_y,param))
    cost_list = np.array(cost_list).reshape(len(intersect),len(gradient))
    
        #Contour plotting in a subplot.
    ax_2d = plt.subplot2grid((2,2), (1,0))
    ax_2d.contourf(intersect, gradient, cost_list, 30, cmap=plt.cm.jet, alpha = 0.5)      
    ax_2d.scatter(optimum[0], optimum[1],lw = 5, color = 'black', edgecolor = 'yellow', 
              marker = '*', label = 'Cost Minimum')
    
        #Line plotting of parameter finding for different learning rates.
    for learning_rate in [0.001, 0.005, 0.05]:
        iteration = 500
        if algo == 'batch':
            cost_hist, param_hist = batch_gd(training_x, training_y, learning_rate, iteration)
        ax_2d.plot(param_hist[:,0], param_hist[:,1], 'o',label = 'learning rate: %.3f' %learning_rate, alpha = 0.6)

        #Label and title setting.
    ax_2d.set_xlabel('$Intersect$', fontsize = 20)
    ax_2d.set_ylabel('$Gradient$', fontsize = 20)
    ax_2d.set_title('Cost Contour Plot', fontsize = 20)
    ax_2d.legend()
    
    
    #Subplot (1,1)
    #Mesh grid of cost.
        #
    plot_count = 0
    for learning_rate in [0.005, 0.05]:
        iteration = 5000
        if algo == 'batch':
            cost_hist, param_hist = batch_gd(training_x, training_y, learning_rate, iteration)
        else: pass

        if plot_count == 0:
            ax[1,1].plot(param_hist[:,0], '-', label = 'Intersect (Learning rate: %.3f)' %learning_rate)
            ax[1,1].plot(param_hist[:,1], '-', label = 'Gradient (Learning rate: %.3f)' %learning_rate)
            plot_count = plot_count + 1           
        elif plot_count != 0:
            ax[1,1].plot(param_hist[:,0], ':', label = 'Intersect (Learning rate: %.3f)' %learning_rate)
            ax[1,1].plot(param_hist[:,1], ':', label = 'Gradient (Learning rate: %.3f)' %learning_rate)
    ax[1,1].legend()
    ax[1,1].set_xlabel('$Iteration$', fontsize = 20)
    ax[1,1].set_ylabel('$Value$', fontsize = 20)
    ax[1,1].set_title('Convergence rate vs learning', fontsize = 20)
```


```python
rng = np.random.RandomState(5)

sample_size = 10
gradient = 5
intersect = 5
training_x = rng.uniform(0, 10, sample_size)
training_y = (gradient*training_x) + (intersect +np.random.normal(0, 1, sample_size))

dm = design_matrix(training_x)

plt.scatter(training_x, training_y)
print('Design matrix: \n', dm)
```

    Design matrix: 
     [[1.         2.21993171]
     [1.         8.70732306]
     [1.         2.06719155]
     [1.         9.18610908]
     [1.         4.88411189]
     [1.         6.11743863]
     [1.         7.65907856]
     [1.         5.18417988]
     [1.         2.96800502]
     [1.         1.87721229]]
    


    
![png](output_9_1.png)
    



```python
algo_summary('batch', training_x, training_y, 100)
```


    
![png](output_10_0.png)
    



```python

```
