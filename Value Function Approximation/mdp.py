import numpy as np
import math
from copy import deepcopy
from typing import List, Dict
# Contains modular functions for MDPs


def value_iteration(S:List,A:List,P:List,R:List,gamma:float,pi:List)->List:
    #Input: ------------------------
    # S: vector of possible states
    # A: vector of all possible actions
    # P: 3-D matrix for every state, probability of going to a next state. shape: P(a,s,s')
    # R: 2-D Matrix for every state, reward of taking every action, shape: R(a,s)
    # gamma: Discount Factor
    # pi: 2-D policy Matrix , shape: pi(s,a)

    #Output:-----------------------
    #best_action: Optimal Policy
    #optimal Value function
    num_states=S.shape[0];
    num_actions=A.shape[0];

    vk=np.zeros([num_states]);
    vk2=np.zeros([num_states]);
    action_vec=np.zeros([num_states])-1;
    assert (R.shape[0]== pi.shape[1]),"R(s,a), pi(a|s) dimension mismatch"
    assert (pi.shape[1]==P.shape[0]),"P(s,a,s'), pi(a|s) dimension mismatch"
    count=0;
    while True:
        
        best_action=np.zeros([num_states])-1;
        for s in range(num_states):
            action_value=float("-inf")
            action_temp=-1;

            if (P[:,s,:]==0).all():
                vk2[s]=0;
                continue

            for a in range(num_actions):
            
                 
                v_temp=R[a,s]+gamma*np.dot(P[a,s,:],vk);
                #print(v_temp,pi[s,a])
                if v_temp > action_value and np.any(P[a,s,:])>0:
                    #print("Entered")
                    action_value=v_temp;
                    action_temp=a;
            vk2[s]=action_value;
            best_action[s]=action_temp
        #print(vk2)
        action_vec=deepcopy(best_action)
        count=count+1;
        if np.all(vk2==vk):
            break;
        else:
            vk=deepcopy(vk2);
            continue;
    return best_action,vk;


def evaluate_policy(S:List,A:List,P:List,R:List,gamma:float,pi:List)->List: # For a particular policy, return value function
    #Input:--------------------------------
    # S: vector of possible states
    # A: vector of all possible actions
    # P: 3-D matrix for every state, probability of going to a next state. shape: P(a,s,s')
    # R: 2-D Matrix for every state, reward of taking every action, shape: R(a,s)
    # gamma: Discount Factor
    # pi: 2-D policy Matrix , shape: pi(s,a)

    # Output:--------------------
    # vk2: Value function for evaluated policy

    num_states=S.shape[0];
    num_actions=A.shape[0];
    
    vk2=np.zeros(S.shape[0]);
    vk=np.zeros(S.shape[0]);
    
    assert (R.shape[0]== pi.shape[1]),"R(s,a), pi(a|s) dimension mismatch"
    assert (pi.shape[1]==P.shape[0]),"P(s,a,s'), pi(a|s) dimension mismatch"
    count=0;
    while True:
        vk=deepcopy(vk2)
        aux=np.zeros([P.shape[1],P.shape[2]])
        for i in range(P.shape[2]):
            aux[:,i]=deepcopy(np.sum(np.multiply(np.transpose(pi),P[:,:,i]),axis=0));
        
        vk2=np.sum(np.multiply(pi,np.transpose(R)),axis=1)+gamma*np.dot(aux,vk);#Upate equation
        count=count+1;
#        print(count)
        if max_diff(vk,vk2)>10**(-6):
            continue;
        else:
            break;
    return vk2

def policy_iteration(S:List,A:List,P:List,R:List,gamma:float,pi:List): # Starting with a certain policy, iterate to optimal policy

    num_states=len(S);
    num_actions=len(A)
    vi=evaluate_policy(S,A,P,R,gamma,pi); # First Policy Evaluation step
    count_iter=0;
    while True : 
        
        #vk=evaluate_policy(S,A,P,R,gamma,pi);
        pi_dash=improve_policy(S,A,P,vi); # Policy Improvement Step;
        vk=evaluate_policy(S,A,P,R,gamma,pi_dash);# Next Policy Evaluation step
        count_iter=count_iter+1;
        if np.all(vk==vi) or count_iter==10000:  # At optimal policy, pi_dash=pi* and vk=v* , will be same as prev iteration
            
            break;
        else:
            vi=deepcopy(vk);
            continue;

    #Map from pi_dash to best action vector
    best_action=np.zeros([num_states])-1;
    
    for s in range(num_states):

        for a in range(num_actions):

            if pi_dash[s,a]==1:

                best_action[s]=a;

    return best_action,vk;

def improve_policy(S:List,A:List,P:List,vk:List): # Policy improvement step
    # Input:----------------------------------
    # S: vector of possible states
    # A: vector of all possible actions
    # P: 3-D matrix for every state, probability of going to a next state. shape: P(a,s,s')
    # R: 2-D Matrix for every state, reward of taking every action. shape: R(a,s)
    # gamma: Discount Factor
    # pi: 2-D policy Matrix , shape: pi(s,a)

    #Output:---------------------------------
    #pi_dash: Improved Policy Matrix: shape(s,a)
    num_states=len(S);
    num_actions=len(A)
    pi_dash=np.zeros([num_states,num_actions]);
    num_states=S.shape[0];
    num_actions=A.shape[0];

    best_action=np.zeros([num_states])-1;

    for s in range(num_states): # 
        pass
        max_valfn=float("-inf")
        #Check which states can be reached from s
        for a in range(num_actions):
                 
            for sd in range(num_states):

                if P[a,s,sd] >0:
                    if vk[sd]> max_valfn:

                        max_valfn=vk[sd];
                        state_bestaction=a
                        
        best_action[s]=state_bestaction

    #convert best action vector to immediate greedy policy
    for s in range(num_states):
        pi_dash[s,int(best_action[s])]=1;

    return pi_dash




def create_MDP(S:List,A:List,P_dict:Dict,R_dict:Dict,gamma:float,pi_dict:Dict): # Given a dictionary for P,R,pi, create relevant matrices
    # Input:--------------------------------
    # S: vector of possible states
    # A: vector of all possible actions
    # P_dict: Nested Dictionary for every state, probability of going to a next state. Format: (s->a->s');
    # R_dict: Nested Dictionary for every state, reward of taking every action. Format(State->Action->Reward);
    # gamma: Discount Factor
    # pi:Nested Dictionary: Format(State->Action->Probability of action);
    
    #Output:--------------------------------
    # S: vector of possible states
    # A: vector of all possible actions
    # P: Matrix of transition Probabilties: shape(a,s,s')
    # R: Matrix of Transition Rewards: shape(a,s);
    # pi: Policy Matrix: shape: (s,a)
    num_states=S.shape[0];
    num_actions=A.shape[0];
    
    R=np.zeros([num_actions,num_states]);
    P=np.zeros([num_actions,num_states,num_states]);
    pi=np.zeros([num_states,num_actions]);

    #Create R

    for key1,val1 in R_dict.items():

        for key2 in val1:

            R[key2,key1]=R_dict[key1][key2];

    #Create P

    for key1,val1 in P_dict.items():

        for key2,val2 in val1.items():

            for key3 in val2:

                P[key2,key1,key3]=P_dict[key1][key2][key3]
    #Create pi

    for key1,val1 in pi_dict.items():

        for key2 in val1:

            pi[key1,key2]=pi_dict[key1][key2];

    return S,A,P,R,gamma,pi

        
def max_diff(a1,a2):

    return  np.max(np.abs(a1-a2))
    

