import numpy as np
import math
from copy import deepcopy
from typing import List, Dict,NewType,Any,Tuple, Callable
from mdp_sampler import mdp_sampler
import os
from sa_features import feature_functions_sa
Array_2D = NewType('Array_2D',List[List[float]]);

def max_diff(a:List,b:List):
    
    return np.max(np.abs(a-b))

def state_actionspace(state:int,action_space:range,sampler)-> List[int]:
    # Gets possible actions for a state, based on the underlying MDP probabilities
    #Inputs:
    #state: The next state
    # Action_space: The total action space of the problem
    #sampler: MDP sampler object
    #Outputs:
    #List of possible actions, subset of action_space
    possible_actions=[]
    for i_a in range(len(action_space)):

        if np.any(sampler.P[action_space[i_a],state,:])>0: # check from underlying MDP is action is possible
            possible_actions.append(action_space[i_a])
    
    if len(possible_actions)==0: # This happens only at terminal state, return any action since Q=0 for any action at terminal

        possible_actions.append(np.random.choice(np.array(action_space)))
    
    return possible_actions

def eps_greedy(w:List,features,action_space:range,next_state:int,eps:float)-> int:

    # epsilon greedy action selection
    # Returns an action according to an epsilon greedy policy
    
    #Inputs:
    # Q: The state-action value function
    # action_space: The set of allowable actions from the next state
    # eps: value of epsilon
    
    #Outputs:
    # Returns an action according to an epsilon greedy policy

    random_selection=np.ones([len(action_space)])*eps/len(action_space)
    prob_vector=np.append(random_selection,[1-eps])  # Random part of policy
   
    Q_select=np.float("-inf");
    
    for i_action in range(len(action_space)):
        
        #Q_temp=np.dot(features[next_state*len(action_space)+i_action,:],w)
        Q_temp=np.dot(features.get_featvector(next_state,i_action),w)
        if Q_temp > Q_select:
            greedy_action=i_action
            Q_select=Q_temp
    #print(greedy_action)
    action_vec=np.append(action_space,[greedy_action]) 
    a= np.random.choice(action_vec,p=prob_vector) # eps-greedy selection
    return a

def sarsa_update(gamma:float,lambd:float,alpha:float,eps:float,ini_state:int,ini_action:int,num_features:int,state_space:range,action_space:range,w:List,E:List,sampler)-> [List,List]:

    # One episode update for SARSA-lambda algorithm with VFA
    # Inputs:---------------
    # gamma: Discount Factor
    #lambd: weighting parameter
    #alpha: learning rate
    #eps: randomness parameter
    #ini_state: The state to start the episode rollout
    #ini_action: the Action to start the episode rollout
    #action space: The entire action space of the MDP
    #w: weights
    #E: Elgibility traces
    #sampler: MDP sampler object, samples next states and rewards
    
    # Outputs:-------------
    #Q: Updated Q value function
    #E Updated elgibility traces
    
    
    features=feature_functions_sa("lagrange",num_features,state_space,action_space)
    current_state=ini_state
    current_action=ini_action
    next_state,some_action,reward=sampler.sample_next(current_state,current_action)

    update_counter=0
    while next_state is not None and gamma**(update_counter)> 10**(-5):
        possible_actions=state_actionspace(next_state,action_space,sampler) # Get space of possible actions for next state(requires underlying MDP)
    
        current_sa_vector=features.get_featvector(current_state,current_action)
        
        next_action=eps_greedy(w,features,possible_actions,next_state,eps)   #Select next action acc to eps-greedy policy
       
        next_sa_vector=features.get_featvector(next_state,next_action)
        
        delta_t=reward+gamma*np.dot(next_sa_vector,w)-np.dot(current_sa_vector,w)
        E=deepcopy(gamma*lambd*E+current_sa_vector)
        dw=deepcopy(alpha*delta_t*E)


        update_counter=update_counter+1
        w=deepcopy(w+dw)
        current_state=next_state
        current_action=next_action # Setup next iterationa and sample from MDP
        next_state,some_action,reward=sampler.sample_next(current_state,current_action) # some_action is not used anywhere, feature of sampler
        #print(next_state,current_state,current_action)
    return [w,E]

def sarsa_lambda_vfa(gamma:float,lambd:float,alpha:float,eps:float,n_episodes:int,state_space:range,num_features:int,action_space:range,sampler)->[List,List] :


    # Main function to be used in other programs; use SARSA_lambda algorithm to get converged state-action value fn(with VFA)

    #Inputs:
    # gamma: Discount Factor
    #lambd: weighting parameter
    #alpha: learning rate
    #eps: randomness parameter
    #n_episodes: Number of episodes to simulate for
    #state_space: The set of visitable states(tabular)
    #action space: The entire action space of the MDP(tabular)
    #sampler: MDP sampler object, samples next states and rewards

    # Outputs:-------------
    #Q_new: Converged Q value function
    #E_new: Converged elgibility traces

    num_states=len(state_space)
    num_actions=len(action_space)
    w=np.ones([num_features])
    E=np.zeros([num_features])
    w_new=np.zeros([num_features])     
    E_new=np.zeros([num_features])
    episode_count=0;

    while episode_count<n_episodes:
        print(episode_count)        
        w=deepcopy(w_new)
        E=deepcopy(E_new)
        ini_state=np.random.choice(state_space)
        ini_action=np.random.choice(action_space)
        [w_new,E_new]=sarsa_update(gamma,lambd,alpha,eps,ini_state,ini_action,num_features,state_space,action_space,w,E,sampler) # One episode of MDP
#        print(w,w_new)
        #print(max_diff(w,w_new)) 
        episode_count=episode_count+1
    return [w_new,E_new]


