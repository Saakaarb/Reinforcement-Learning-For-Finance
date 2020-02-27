import numpy as np
import math
from copy import deepcopy
from typing import List, Dict,NewType,Any,Tuple

Array_2D = NewType('Array_2D',List[List[float]]);
Array_3D = NewType('Array_3D',List[List[List[float]]]);



class mdp_sampler: # Takes the matrix structure of an MDP, and returns samples according to defined probabilities

    def __init__(

            self,
            state_space:range,     # Specify States using indice
            action_space:range,   # Specify actions using indices
            P:Array_3D,           # P(s'|s,a): format is P(a,s,s')
            pi:Array_2D,          # pi(a|s): format is pi(s,a)
            R:Array_2D)-> None: # Reward function: shape: R(a,s)
        self.state_space=state_space
        self.action_space=action_space
        self.P=P
        self.pi=pi
        self.R=R

    
    def sample_next(self,state_index:int,action_index:int)-> [int,int,float]:
        
        if sum(self.P[action_index,state_index,:]) ==0:
            return None,None,self.R[action_index,state_index]

        next_state_index=np.random.choice(self.state_space,p=self.P[action_index,state_index,:])
        next_action_index=np.random.choice(self.action_space,p=self.pi[state_index,:])
        next_reward=self.R[action_index,state_index]

        return next_state_index,next_action_index,next_reward
        




        
