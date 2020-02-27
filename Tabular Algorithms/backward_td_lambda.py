import numpy as np
import math
from copy import deepcopy
from typing import List, Dict,NewType,Any,Tuple, Callable
from mdp_sampler import mdp_sampler

Array_2D = NewType('Array_2D',List[List[float]]);
Array_3D = NewType('Array_3D',List[List[List[float]]]);

# Evaluation of value function
def bwd_td_lambda_update_offline(gamma:float,lambd:float,alpha:float,ini_state:int,ini_action:int,state_space:range,V:List[float],E:List[float],sampler:Callable)-> [List[float],List[float]]:

    # Update V for ONE EPISODE, until termination or convergence

    #V_new=np.zeros([len(state_space)])
    
    next_state,next_action,reward=sampler.sample_next(ini_state,ini_action); # get values from sampler,actual MDP hidden from function
    current_state=ini_state
    update_counter=0
    accumulated_change=np.zeros([len(state_space)])
    #print(next_state,next_action,reward)
    while (next_state is not None and next_action is not None) and gamma**(update_counter)>10**(-5) : # None for either implies terminal state
        
        delta_t=reward+gamma*V[next_state]-V[current_state]
        for i_E in range(len(E)):

            if i_E==current_state:
                increment=1
            else:
                increment=0
            E[i_E]=gamma*lambd*E[i_E]+increment
        accumulated_change=accumulated_change+alpha*delta_t*E
        update_counter=update_counter+1
       
        current_state=next_state
        current_action=next_action
     #   print("Looped")
        next_state,next_action,reward=sampler.sample_next(current_state,current_action)
    
    V_new=np.array(V)+accumulated_change
    return [V_new,E]


def bwd_td_lambda_update_online(gamma:float,lambd:float,alpha:float,ini_state:int,ini_action:int,state_space:range,V:List[float],E:List[float],sampler:Callable)-> [List[float],List[float]]:

    # Update V for ONE EPISODE, until termination or convergence

    V_new=np.zeros([len(state_space)])

    next_state,next_action,reward=sampler.sample_next(ini_state,ini_action); # get values from sampler,actual MDP hidden from function
    current_state=ini_state
    update_counter=0
    #print(next_state,next_action,reward)
    while (next_state is not None and next_action is not None) and gamma**(update_counter)>10**(-5) : # None for either implies terminal state

        delta_t=reward+gamma*V[next_state]-V[current_state]
        for i_E in range(len(E)):

            if i_E==current_state:
                increment=1
            else:
                increment=0
            E[i_E]=gamma*lambd*E[i_E]+increment


        for i_state in range(len(state_space)):

            V_new[i_state]=V[i_state]+alpha*delta_t*E[i_state]
        update_counter=update_counter+1

        current_state=next_state
        current_action=next_action
     #   print("Looped")
        next_state,next_action,reward=sampler.sample_next(current_state,current_action)
        V=deepcopy(V_new)
    return [V,E]




def bwd_td_lambda(gamma:float,lambd:float,alpha:float,n_episodes:int,state_space:range,action_space:range,terminal_states:List[int],sampler:Callable,mode="online")-> List[float]:

    num_states=len(state_space)
    # Initialize V
    V=np.zeros([num_states])
    #Initialize E
    E=np.zeros([num_states])

    i_episodes=0;
    while i_episodes < n_episodes:
        #print(i_episodes)
        ini_state=np.random.choice(state_space)
        ini_action=np.random.choice(action_space)
        if mode=="online":
            [V_new,E_new]=bwd_td_lambda_update_online(gamma,lambd,alpha,ini_state,ini_action,state_space,V,E,sampler)
        elif mode=="offline":
            [V_new,E_new]=bwd_td_lambda_update_offline(gamma,lambd,alpha,ini_state,ini_action,state_space,V,E,sampler)
        else:
            raise Exception("Select Valid Mode")
        i_episodes=i_episodes+1
        V=deepcopy(V_new)
        E=deepcopy(E_new)

    return [V_new,E_new]

def max_diff(a1,a2):

    return  np.max(np.abs(a1-a2))

