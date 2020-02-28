import numpy as np
import math
from copy import deepcopy
from typing import List, Dict,NewType,Any,Tuple, Callable
from mdp_sampler import mdp_sampler
import os
Array_2D = NewType('Array_2D',List[List[float]]);


def state_actionspace(state,action_space,sampler):

    # Gets possible actions for a state, based on the underlying MDP probabilities
    #Inputs:
    #state: The next state
    # Action_space: The total action space of the problem
    #sampler: MDP sampler object
    #Outputs:
    #List of possible actions, subset of action_space

    possible_actions=[]
    for i_a in range(len(action_space)):

        if np.any(sampler.P[action_space[i_a],state,:])>0: # check from underlying MDP if action is possible
            possible_actions.append(action_space[i_a])
    
    if len(possible_actions)==0: # This happens only at terminal state, return any action since Q=0 for any action at terminal

        possible_actions.append(np.random.choice(np.array(action_space)))
    
    return possible_actions

def eps_greedy(Q:Array_2D,action_space:range,next_state:int,eps:float)-> int:

    # epsilon greedy action selection
    # Returns an action according to an epsilon greedy policy

    #Inputs:
    # Q: The state-action value function
    # action_space: The set of allowable actions from the next state
    #next_state: state index
    # eps: value of epsilon

    #Outputs:
    # Returns an action according to an epsilon greedy policy

    random_selection=np.ones([len(action_space)])*eps/len(action_space)
    prob_vector=np.append(random_selection,[1-eps])
    greedy_actions=np.where(Q[next_state,:]==np.max(np.take(Q[next_state,:],action_space)))

    for i in range(len(greedy_actions[0])):
        if greedy_actions[0][i] in action_space:
            final_action=greedy_actions[0][i]
    action_vec=np.append(action_space,[final_action]) # if 
    return np.random.choice(action_vec,p=prob_vector)


def greedy(Q:Array_2D,action_space:range,next_state:int,eps:float)-> int:

    #  greedy action selection
    # Returns an action according to an epsilon greedy policy

    #Inputs:
    # Q: The state-action value function
    # action_space: The set of allowable actions from the next state
    # next_state: state index
    # eps: value of epsilon

    #Outputs:
    # Returns an action according to an epsilon greedy policy

    greedy_actions=np.where(Q[next_state,:]==np.max(np.take(Q[next_state,:],action_space)))

    for i in range(len(greedy_actions[0])):
        if greedy_actions[0][i] in action_space:
            final_action=greedy_actions[0][i]
    
    return final_action


def q_update(gamma:float,lambd:float,alpha:float,eps:float,ini_state,ini_action,action_space,Q:Array_2D,E:Array_2D,sampler)-> [Array_2D,Array_2D]:

    # One episode update for Q-lambda algorithm
    # Inputs:-------------------
    # gamma: Discount Factor
    #lambd: weighting parameter
    #alpha: learning rate
    #eps: randomness parameter
    #ini_state: The state to start the episode rollout
    #ini_action: the Action to start the episode rollout
    #action space: The entire action space of the MDP
    #Q: State-Action value function
    #E: Elgibility traces
    #sampler: MDP sampler object, samples next states and rewards

    # Outputs:-------------
    #Q: Updated Q value function
    #E Updated elgibility traces


    num_states=Q.shape[0]
    num_actions=Q.shape[1]

    current_state=ini_state
    current_action=ini_action
    next_state,some_action,reward=sampler.sample_next(current_state,current_action)
    Q_new=np.zeros([num_states,num_actions])
    update_counter=0
    while next_state is not None and gamma**(update_counter)> 10**(-5):
        possible_actions=state_actionspace(next_state,action_space,sampler)
        next_action_update=greedy(Q,possible_actions,next_state,eps)                           
        delta_t=reward+gamma*Q[next_state,next_action_update]-Q[current_state,current_action]
        E[current_state,current_action]=E[current_state,current_action]+1
        for i_state in range(num_states):

            for i_action in range(num_actions):
                Q_new[i_state,i_action]=Q[i_state,i_action]+alpha*delta_t*E[i_state,i_action]
                E[i_state,i_action]=lambd*gamma*E[i_state,i_action]
        update_counter=update_counter+1
        next_action=eps_greedy(Q,possible_actions,next_state,eps)
        Q=deepcopy(Q_new)
        current_state=next_state
        current_action=next_action
        next_state,some_action,reward=sampler.sample_next(current_state,current_action) # some_action is not used anywhere, feature of sampler
    return [Q,E]

def q_lambda(gamma:float,lambd:float,alpha:float,eps:float,n_episodes:int,state_space:range,action_space:range,sampler)->[Array_2D,Array_2D] :

    # Main function to be used in other programs; use q_lambda algorithm to get converged state-action value fn

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
    Q=np.zeros([num_states,num_actions])
    E=np.zeros([num_states,num_actions])
    
    episode_count=0;

    while episode_count<n_episodes:
        
        ini_state=np.random.choice(state_space)
        ini_action=np.random.choice(action_space)
        [Q_new,E_new]=q_update(gamma,lambd,alpha,eps,ini_state,ini_action,action_space,Q,E,sampler)
        
        Q=deepcopy(Q_new)
        E=deepcopy(E_new)
        episode_count=episode_count+1
    return [Q_new,E_new]


