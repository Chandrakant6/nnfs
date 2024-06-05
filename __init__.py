import numpy as np

from .dataset import vertical_data

class Layer:
    def __init__(self, n_in, n_neu):
        self.wt = .1 * np.random.randn(n_in, n_neu)
        self.b = np.zeros((1,n_neu))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.wt) + self.b

class ReLU: #for layers
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        
class SoftMax: #for output
    def  forward(self, inputs):
        exp_val = np.exp(inputs - np.max(inputs, 
                axis=1, keepdim=True)) #exponent
        prob = exp_val / np.sum(exp_val, 
                axis=1 ,keepdim=True) #normalize

class Loss:
    def calculate(self, inputs, y):
        sample_loss = self.forward(inputs, y)
        data_loss = mp.mean(sample_loss)
        return data_loss

class CCE: # categorical cross entropy loss
    def calculate(self, y_pred, y_true):
        sample = len(y_pred)
        clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            confidence = clipped(range(sample), y_true)
        else:
            confidence = np.sum(clipped * y_true, axis=1)

        return np.log(confidence) #negative log likelihood
