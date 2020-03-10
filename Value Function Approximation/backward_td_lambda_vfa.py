import numpy as np
import math
from copy import deepcopy
from typing import List, Dict,NewType,Any,Tuple, Callable
from mdp_sampler import mdp_sampler
from feature_functions import feature_functions

Array_2D = NewType('Array_2D',List[List[float]]);
Array_3D = NewType('Array_3D',List[List[List[float]]]);

def isNaN(num):
    return num != num



def bwd_td_lambda_update_online(gamma:float,lambd:float,alpha:float,ini_state:int,ini_action:int,state_space:range,num_features:int,w:List[float],E:List[float],sampler:Callable)-> [List[float],List[float]]:

    # Update V for ONE EPISODE, until termination or convergence

    #Inputs:-----------
    #gamma: Discount factor
    #lambd: weighting param
    #alpha: Learning rate
    #ini_state: Initil state of episode
    #ini_action: Initial action of episode
    #num_features: number of feature functions to use
    #w: current weights
    #E: current eligibility traces
    #sampler:sampler object containing info about underlying MRP

    #Outputs:
    #w: updated weights after 1 episode
    #e: eligibility traces at and of episode 


    w_new=np.zeros([len(w)])
   
    features=feature_functions("lagrange",num_features,state_space)


    next_state,next_action,reward=sampler.sample_next(ini_state,ini_action); # get values from sampler,actual MDP hidden from function
    current_state=ini_state
    update_counter=0
    while (next_state is not None and next_action is not None) and gamma**(update_counter)>10**(-5) : # None for either implies terminal state
        currstate_features=features.get_featvector(current_state);
        nextstate_features=features.get_featvector(next_state);
        delta_t=reward+gamma*np.dot(w,nextstate_features)-np.dot(w,currstate_features)
        #print(np.dot(w,nextstate_features))
        if isNaN(np.dot(w,nextstate_features)) :
            raise Exception(" Value function is NaN")

        E=deepcopy(gamma*lambd*E+currstate_features)
        #print(currstate_features)
        dw=deepcopy(alpha*delta_t*E)
        w_new=deepcopy(w+dw)

        update_counter=update_counter+1

        current_state=next_state
        current_action=next_action
     #   print("Looped")
        next_state,next_action,reward=sampler.sample_next(current_state,current_action)
        w=deepcopy(w_new)
    return [w,E]




def bwd_td_lambda_vfa(gamma:float,lambd:float,alpha:float,n_episodes:int,state_space:range,num_features,action_space:range,sampler:Callable,mode="online")-> [List[float],List[float]]:
    

    # Main function to be used; Backward TD-Lambda with Value function approximation
    #Inputs:-----------
    #gamma: Discount factor
    #lambd: weighting param
    #alpha: Learning rate
    #n_episodes: Number of episodes to use
    #state space: description of state space, a range
    #num_features: number of feature functions to use
    #action space: List describing action space
    #sampler: Sampler object containing info about MRP
    #mode: online or offline

    #Outputs:
    #w_new: Final weights after n_episodes
    #E: eligibility traces at the end

    num_states=len(state_space)
    # Initialize V
    w=np.zeros([num_features])
    #Initialize E
    E=np.zeros([num_features])

    i_episodes=0;
    while i_episodes < n_episodes:
        print(i_episodes)
        ini_state=np.random.choice(state_space)
        ini_action=np.random.choice(action_space)
        if mode=="online":
            [w_new,E_new]=bwd_td_lambda_update_online(gamma,lambd,alpha,ini_state,ini_action,state_space,num_features,w,E,sampler)
        i_episodes=i_episodes+1
        w=deepcopy(w_new)
        E=deepcopy(E_new)

    return [w_new,E_new]

def max_diff(a1,a2):

    return  np.max(np.abs(a1-a2))

