#
# Defines state-action features functions for MDPs
# Can choose between lagrange, polynomial(powers of x) or laguerre 
import numpy as np
import math
from typing import List
from copy import deepcopy
from Basis import EvalBasis
def laguerre_vals(state):

    f1=1
    f2=np.exp(-state/2)
    f3=np.exp(-state/2)*(1-state)
    #f4=np.exp(-state/2)*(1-2*state+state**2/2)
    return np.array([f1,f2,f3])

class feature_functions_sa:

    def __init__(

            self,
            name:str,
            num_features:int,
            state_space:range,
            action_space:range) -> None:

        self.name=name
        self.num_features=num_features
        self.state_space=state_space
        self.action_space=action_space

        if self.name=="lagrange":

            feature_vals=np.zeros([len(self.state_space)*len(self.action_space),self.num_features])
            xmax=self.state_space[-1]
            xmin=self.state_space[0]
            basis_state=EvalBasis(self.num_features-1, len(self.state_space), self.state_space,xmin,xmax)
            xmax=self.action_space[-1]
            xmin=self.action_space[0]
            basis_action=EvalBasis(self.num_features-1,len(self.action_space),self.action_space,xmin,xmax)

            for i_features in range(num_features):

                for j_features in range(len(self.state_space)):

                    for k_features in range(len(self.action_space)):

                        feature_vals[j_features*len(self.action_space)+k_features,i_features]=basis_state[j_features,i_features]+basis_action[k_features,i_features]

        if self.name=="polynomial":

            feature_vals=np.zeros([len(self.state_space)*len(self.action_space),self.num_features])

            for i_features in range(self.num_features):

                for j_features in range(len(self.state_space)):

                    for k_features in range(len(self.action_space)):
                        feature_vals[j_features*len(self.action_space)+k_features,i_features]=((j_features+1)/len(self.state_space))**(i_features+1)+((k_features+1)/len(self.action_space))**(i_features+1)

        if self.name=="laguerre":

            feature_vals=np.zeros([len(self.state_space),self.num_features])

            for j_features in range(len(self.state_space)):

                feature_vals[j_features,:]=deepcopy(laguerre_vals(j_features/len(self.state_space)))

        self.feature_vals=feature_vals


    def get_featvector(self,state,action):
        
        return self.feature_vals[state*len(self.action_space)+action,:]
