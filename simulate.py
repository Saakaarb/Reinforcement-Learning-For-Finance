import numpy as np
import math
import data_structures

def solve_MRP():
    #Solution of MRP discussed in class for gamma !=1 by matrix inversion method
    MRP_ex=data_structures.MRP();
    gamma=MRP_ex.gamma;
    P_s_sd=MRP_ex.P_s_sd;
    rewards=MRP_ex.rewards;

    V=np.matmul(np.linalg.inv(np.identity(7)-gamma*P_s_sd),np.transpose(rewards))

    print(V)

def reward_func():

    MRP_ex=data_structures.MRP()

    return np.matmul(MRP_ex.P_s_sd,np.transpose(MRP_ex.rewards));


#--------------

#Solution of MDP discussed in class by matrix inversion method
def solve_MDP():
    MDP_ex=data_structures.MDP()

    gamma=MDP_ex.gamma;
    P=MDP_ex.P_a_s_sd;
    rewards=np.transpose(MDP_ex.rewards);
    n_states=MDP_ex.num_states;
    policy=np.transpose(MDP_ex.policy);

    V=np.matmul(np.linalg.inv(np.identity(n_states)-gamma*np.matmul(policy,P)),np.matmul(policy,rewards))

    print(V)

