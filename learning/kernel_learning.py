import numpy as np
import copy

from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

#from sklearn.kernel_ridge import KernelRidge

class SafetyFilteringModel():
    def __init__(self, training_data):
        self.input_states = np.array( training_data['input_states'] ) 
        self.targets_V = np.array( training_data['targets_V'] ) 
        self.targets_V = self.targets_V[:, np.newaxis]
        self.targets_Vgrad = np.array(training_data['targets_Vgrad'] )
        self.final_targets = np.concatenate((self.targets_V, self.targets_Vgrad), axis=1)
        self.state_dim = self.input_states.shape[1]
        self.kernel_Vgrad = RBF(length_scale=[1.0, 1.0, 1.0, 1.0, 1.0], length_scale_bounds=(1.0, 1e4))
        self.kernel_V = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e4))
        self.model_Vgrad = GaussianProcessRegressor(kernel=self.kernel_Vgrad, alpha=1e-3)
        self.model_V = GaussianProcessRegressor(kernel=self.kernel_V, alpha=1e-3)

    def train_model(self):
        #self.scaler = preprocessing.StandardScaler().fit(self.input_states)
        #self.states_scaled = self.scaler.transform(self.input_states)
        self.model_Vgrad.fit(self.input_states, self.targets_Vgrad)
        self.model_V.fit(self.input_states, self.targets_V)
        print(self.model_Vgrad.kernel.length_scale)
        print(self.model_V.kernel.length_scale)

    def predict_model(self, input):
        #input_transform = self.scaler.transform(input)
        Vgrad_predict, Vgrad_variance = self.model_Vgrad.predict(input, return_std=True)
        V_predict, V_variance = self.model_V.predict(input, return_std=True)
        #print(V_predict, Vgrad_predict, Vgrad_variance, V_variance)
        return V_predict, Vgrad_predict