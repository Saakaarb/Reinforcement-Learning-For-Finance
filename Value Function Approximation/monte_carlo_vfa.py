#
# Monte-carlo prediction algorithm using value function approximation
#
import numpy as np
import math
from mdp_sampler import mdp_sampler
from feature_functions import feature_functions
from typing import Callable,List
from copy import deepcopy
def mc_update(gamma:float,alpha:float,ini_state:int,ini_action:int,state_space:range,num_features:int,w:List[float],sampler)-> List[float]:

    # Completes one episode and makes updates to weights
    features=feature_functions("lagrange",num_features,state_space)

    next_state,next_action,reward=sampler.sample_next(ini_state,ini_action); # get values from sampler,actual MDP hidden from function
    current_state=ini_state
    update_counter=0
    Gt=0;
    currstate_features=features.get_featvector(current_state); 
    # Calculate Return of a finite horizon simulation
    while (next_state is not None and next_action is not None) and gamma**(update_counter)>10**(-5) : # None for either implies terminal state
        Gt=Gt+reward
        currstate_features=features.get_featvector(current_state);
        update_counter=update_counter+1
        current_state=next_state
        current_action=next_action
        next_state,next_action,reward=sampler.sample_next(current_state,current_action) # Following policy pi current a gives r,s'
    
    #print(np.dot(currstate_features,w))
    dw=alpha*(Gt-np.dot(currstate_features,w))*currstate_features

    w=w+dw
    return w

def monte_carlo(gamma:float,alpha:float,n_episodes:int,state_space:range,num_features:int,action_space:range,sampler:Callable):
    
    #Obtains weights after n episodes using monte-carlo algorithm
    num_states=len(state_space)
    # Initialize V
    w=np.zeros([num_features])

    i_episodes=0;
    while i_episodes < n_episodes:
        print(i_episodes)
        ini_state=np.random.choice(state_space)
        ini_action=np.random.choice(action_space)
        
        w_new=mc_update(gamma,alpha,ini_state,ini_action,state_space,num_features,w,sampler)
        i_episodes=i_episodes+1
        w=deepcopy(w_new)
    
    return [w_new]


