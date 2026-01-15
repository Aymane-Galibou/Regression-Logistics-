import numpy as np


class LogisticRegression():

    # initializing attributes 
    def __init__(self,learning_rate,no_of_iteration):
        self.learning_rate=learning_rate
        self.no_of_iteration=no_of_iteration

    # the sigmoid function
    def sigmoid(self,Z):
        return 1/(1+np.exp(-np.clip(Z,-250,250)))
    
    # fit function for model learning 
    def fit(self,X,Y):

        # Transformin X and Y from dataframe to numpy arrays
        X=np.array(X)
        Y=np.array(Y)

        # number of training example and number of feature  
        self.m,self.n=X.shape

        # initializing the weight and bias 
        self.w=np.zeros(self.n)
        self.b=0
        self.X=X
        self.Y=Y
        
        # applying gradient descent
        for i in range(self.no_of_iteration):
            self.update_weights()

    # defining the gradient descent function
    def update_weights(self):
        Z=self.X.dot(self.w) + self.b
        Y_cap=self.sigmoid(Z)

        # handling the weight 
        dw= ( (self.X.T).dot(Y_cap-self.Y ) ) / self.m
        self.w-= self.learning_rate * dw

        # handling the bias 
        db = np.sum(( Y_cap - self.Y )) / self.m
        self.b -= self.learning_rate*db

    
    # the predict function for logistic regression
    def predict(self,X):
        Z=X.dot(self.w) + self.b
        Y_cap=self.sigmoid(Z)
        Y_cap=np.where( Y_cap >= 0.5,1,0)
        return Y_cap







