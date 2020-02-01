### Linear regression based on QR decomposition
Gram-Schmidt Method is used for implementation of QR decomposition.

If we consider our formula as :

y = b*X
    
### Parameters
X  : 1d array
        input values

y : 1d array
        considered output values of the model


### Example

```python
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
	
'''

### Notes
   
- If you do not have good value for MSE and RMSE, try to add more point to X and y based on your equation or system to help the model for finding a better model.
Please pay attention that adding irrelevant data will lead to a worse model. 
    
- This model is a one-dimensional polynomial equation
   
