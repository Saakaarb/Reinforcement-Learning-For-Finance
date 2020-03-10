#
# Defines state-feature functions for MRPs
# Can choose between polynomial, lagrange or laguerre functions
import numpy as np
import math
from typing import List
from copy import deepcopy
from Basis import EvalBasis
from matplotlib import pyplot as plt
def laguerre_vals(state):

    f1=1
    f2=np.exp(-state/2)
    f3=np.exp(-state/2)*(1-state)
    #f4=np.exp(-state/2)*(1-2*state+state**2/2)
    return np.array([f1,f2,f3])

class feature_functions:

    def __init__(

            self,
            name:str,
            num_features:int,
            state_space:range) -> None:

        self.name=name
        self.num_features=num_features
        self.state_space=state_space
        
        if self.name=="lagrange":

            #feature_vals=np.zeros([len(self.state_space),self.num_features])
            xmax=self.state_space[-1]
            xmin=self.state_space[0]
            feature_vals=EvalBasis(self.num_features-1, len(self.state_space), self.state_space,xmin,xmax)
            plt.plot(self.state_space,feature_vals[:,0])
            plt.plot(self.state_space,feature_vals[:,1])
            plt.plot(self.state_space,feature_vals[:,2])
            plt.title("Lagrange Polynomials for p=2")
            plt.xlabel("State")
            plt.ylabel("Value")
            plt.show()
        
        if self.name=="polynomial":

            feature_vals=np.zeros([len(self.state_space),self.num_features])

            for i_features in range(self.num_features):

                for j_features in range(len(self.state_space)):
                    feature_vals[j_features,i_features]=((j_features+1)/len(self.state_space))**(i_features)

        if self.name=="laguerre":

            feature_vals=np.zeros([len(self.state_space),self.num_features])

            for j_features in range(len(self.state_space)):

                feature_vals[j_features,:]=deepcopy(laguerre_vals(j_features/len(self.state_space)))

        self.feature_vals=feature_vals


    def get_featvector(self,state):
        
        return self.feature_vals[state,:]
