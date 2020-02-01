# -*- coding: utf-8 -*-
"""
@author: Kazem Gheysari

kgheysari@gmail.com

"""

import numpy as np


class LinearRegression:
    '''
    Linear regression based on QR decomposition
    
    
    If we consider our formula as :
        y = b*X
    
    Parameters
    ----------
    X : 1d array
        input values for X
    y : 1d array
        considered output values of the model
        
    Example
    -------        
    # Training input data
	X = np.array([
    [1],
    [2],
    [-1]])            
	
    # Training target data
    y = np.array([
    [0],
    [1],
    [-2]])  
    
	# Create a model
    mdl = LinearRegression()
    
	# Train model with training data
    b = mdl.fit(X,y)
    
    # Inut test for testing the model with new values
    Xtest = np.array([
    [3],
    [-2]])  
    
    # Target values of the input test
    Ytarget = np.array([
    [2],
    [-3]])  
	
    # Apply test value to the model
    Ytest = mdl.predict(Xtest)
	
    # MSE,RMSE Calculation
    MSE,RMSE = mdl.model_accuray(Ytarget, Ytest)     

    Notes
    -------        
    If you do not have good value for MSE and RMSE, try to add more point to X and y
    
    This model in a linear model or one-dimensional polynomial equation
    
    '''
    
    
    def __init__(self):
        self.b =0
        self.yhat = 0
        self.A =[]

    def fit(self, X, y): #-> LinearRegression: X is Tuple[float]
        Q, R = self.QR_decomp(X)
        d = np.dot(Q.transpose(),y)
        b = np.dot(np.linalg.inv(R),d)
        self.b = b
        return b
    
    def predict(self, X): 
        b = self.b
        yhat = np.dot(X,b)
        self.yhat = yhat
        return yhat
    
    def model_accuray(self,Ytarget,Yhat):
        '''
        It is used for calculation of the accuracy of the regression model
        
        Parameters
        ----------
        Ytarget : 1d array
            Real output of the test input
        Yhat : 1d array
            output of the model

        Returns
        -------
        MSE  :  float, scalar
            Mean Square Error.
        RMSE :  floar, scalar
            Root Mean Square Error.            
    
    
        Notes
        -------        
        Low values for MSE and RMSE shows a better model
        Ideal value for MSE and RMSE is zero

        '''
        
        error = Ytarget - Yhat
        SqrError = [e*e for e in error]
        SumSqrError = sum(SqrError)
        MSE = SumSqrError/max(error.shape)
        RMSE = np.sqrt(MSE)
        return float(MSE),float(RMSE)
    
    def norm_vector(self,n):
        
        '''
        This function claculate the norm of input vector
        
        
        Parameters
        ----------
        n : 1d array
            Input vector

        Returns
        -------
        Norm : float, scalar
            Norm of input vector
          
        
        Example
        -------
            norm_vector([25, 36, 9])
            
        returns    
            6.324555320336759    
    
        Notes
        -------        
        It is used for QR decomposition
        
        '''
        n2 = [x*x for x in n]
        sum_n2 = sum(n2)
        return np.sqrt(sum_n2)
    
    def QR_decomp(self,A):
        '''
        QR decomposition of Matrix A using Gram-Schmidt Method
        
        
        Parameters
        ----------
        A : 2d array, m*n array
            Input matrix

        Returns
        -------
        Q : float, matrix
            orthogonal matrix Q
            
        R : float, mtrix
            upper triangular matrix R
       
        
        
        example :
            A = np.array([[3, 6,7],
                        [4, 5,9],])
            
            Q, R =  QR_decomp (A) 
    
        Notes
        -------        
        It is used for QR decomposition
        
        '''        
        
        m,n = A.shape         
        
        z = np.zeros((n,1))
        Q = np.zeros((m,n))
        R = np.zeros((n,n))        
        
        for i in range(m):
            Q[i,0] = np.divide(A[i,0], self.norm_vector(A[:,0]))
            
        sum_s =[]
        for i in range(1,n):
            sum_s = np.zeros((1,m))
            for k in range(0,i):
                sum_s = sum_s + Q[:,k]*np.dot(Q[:,k],A[:,i].reshape((-1, 1)))        
                z = A[:,i] - sum_s
                
            z = z[0]
            
            for j in range(m):
                Q[j,i] = z[j]/self.norm_vector(z)  
            
        for i in range(m):
            for j in range(i,n):
                R[i,j] = np.dot(A[:,j],Q[:,i].reshape((-1, 1)))
        
        return Q,R

if __name__ == '__main__':
    
    # Training input data
    X = np.array([
    [1],
    [2],
    [-1]])   
         
    # Training target data
    y = np.array([
    [0],
    [1],
    [-2]])  
    
    # Create a model
    mdl = LinearRegression()
    
    # Train model with training data
    b = mdl.fit(X,y)
    
    # Inut test for testing the model with new values
    Xtest = np.array([
    [3],
    [-2]])  
    
    # Target values of the input test
    Ytarget = np.array([
    [2],
    [-3]])  
    
    # Apply test value to the model
    Ytest = mdl.predict(Xtest)
    
    # MSE,RMSE Calculation
    MSE,RMSE = mdl.model_accuray(Ytarget, Ytest)      

