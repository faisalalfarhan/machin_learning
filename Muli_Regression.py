import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression as slr

# from sklearn.datasets import make_regression
# X, Y = make_regression(n_samples=100, n_features=1, n_targets=1, noise=20)

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost

#Read dataset and see shape of X
path='F:\\Faisal//m_r.txt'
df = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
X = df.iloc[:,:-1].values
Y = df.iloc[:,1].values
print(X.shape)
#print(df.describe())

# rescaling data (beacuse the data range is very different (نعرفه من المتوسط))
df = (df - df.mean()) / df.std()
print('data after normalization = ', df.head(10))

#add ones column
df.insert(0, 'Ones', 1)

# separate X (training data) from y (target variable)
cols = df.shape[1]
X2 = df.iloc[:,0:cols-1]
y2 = df.iloc[:,cols-1:cols]

print('X2 data = \n' ,X2.head(10) )
print('y2 data = \n' ,y2.head(10) )


# convert to matrices and initialize theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))


print('X2 \n',X2)
print('X2.shape = ' , X2.shape)
print('**************************************')
print('theta2 \n',theta2)
print('theta2.shape = ' , theta2.shape)
print('**************************************')
print('y2 \n',y2)
print('y2.shape = ' , y2.shape)
print('**************************************')

# initialize variables for learning rate and iterations
alpha = 0.1
iters = 100

# perform linear regression on the data set
g2, cost2 = gradientDescent (X2, y2, theta2, alpha, iters)

# get the cost (error) of the model
thiscost = computeCost(X2, y2, g2)


print('g2 = ' , g2)
print('cost2  = ' , cost2[0:50] )
print('computeCost = ' , thiscost)
print('**************************************')
