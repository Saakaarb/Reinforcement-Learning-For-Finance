import numpy as np


def create_mrp_Rs_sd(S,P_dict,R_dict,gamma): # R(s,s') definition
    # S: vector of all possible states
    # P_dict: dictionary containing P(s->s') info
    # R_dict: dictionary containing R(s->s') info
    # gamma: discount factor
    num_states=S.shape[0];

    R=np.zeros([num_actions,num_states]);
    P=np.zeros([num_actions,num_states,num_states]);

    #Create R

    for key1,val1 in R_dict.items():

        for key2 in val1:

            R[key1,key2]=R_dict[key1][key2];

    #Create P

    for key1,val1 in P_dict.items():

        for key2,val2 in val1.items():

            P[key1,key2]=P_dict[key1][key2][key3]

    return S,P,R,gamma

def create_mrp_Rs(S,P_dict,R,gamma):

    # S: vector of all possible states
    # P_dict: dictionary containing P(s->s') info
    # R: vector containing R(s) info
    # gamma: discount factor
    num_states=S.shape[0];

    P=np.zeros([num_actions,num_states,num_states]);

    #Create P

    for key1,val1 in P_dict.items():

        for key2,val2 in val1.items():

            P[key1,key2]=P_dict[key1][key2][key3]

    return S,P,R,gamma



def change_reward_def(S,P,R): #Convert from R(s,s') to R(s)
    
    num_states=S.shape[0];
    assert (np.all(R.shape==[num_states,num_states])),'R shape incorrect!'
    assert (np.all(P.shape==[num_states,num_states])),'P shape incorrect!'
    
    R_s=np.sum(np.multiply(R,P),axis=1)

    return R_s



def mdp_to_mrp(S,A,P,R,gamma,pi):# Convert an MDP to MRP for a given policy pi
    
    
    P_pi=np.zeros([P.shape[1],P.shape[2]])
    for i in range(P.shape[2]):
        P_pi[:,i]=deepcopy(np.sum(np.multiply(np.transpose(pi),P[:,:,i]),axis=0));

    R_pi=deepcopy(np.sum(np.multiply(pi,np.transpose(R)),axis=1))
    #vk2=np.sum(np.multiply(pi,np.transpose(R)),axis=1)+gamma*np.dot(aux,vk);#Upate equation
    return (S,A,P_pi,R_pi,gamma)


def solve_MRP(S,P,R_s,gamma):# Solve the MRP using matrix inversion
    
    num_states=S.shape[0];    

    if np.all(R_s.shape==[num_states,num_states]):
        R_s=deepcopy(change_rewar_def(S,R_s,P));
    
    if np.all(R_s.shape==[1,num_states]):
        R_s=deepcopy(np.transpose(R_s));

    V=np.matmul(np.linalg.inv(np.identity(num_states)-gamma*P),R_s)

    return V


