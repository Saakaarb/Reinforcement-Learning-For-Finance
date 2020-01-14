import numpy as np
import math


class MRP: # data structure for student MRP discussed in slides

    num_states=7;
    gamma=0.9;
    initial_state=1;

    P_s_sd=np.zeros([num_states,num_states]);
    r_s_sd=np.zeros([num_states,num_states]);# r_s_s' definition of reward

    states=range(num_states);
    rewards=np.array([-1,-2,-2,-2,10,1,0]);# R_s definition of reward(currently using values from lecture)
    #-----------Fill in Probability distribution

    P_s_sd[0,0]=0.9;
    P_s_sd[0,1]=0.1;
    P_s_sd[1,0]=0.5;
    P_s_sd[1,2]=0.5;
    P_s_sd[2,3]=0.8;
    P_s_sd[2,6]=0.2;
    P_s_sd[3,4]=0.6;
    P_s_sd[3,5]=0.4;
    P_s_sd[5,1]=0.2;
    P_s_sd[5,2]=0.4;
    P_s_sd[5,3]=0.4;
    P_s_sd[4,6]=1;
    P_s_sd[6,6]=1;
    #-----------------------------


class MDP: # data structure for student MDP discussed in slides

    num_states=6;
    actions=['FB','Quit','Study','Pub','Sleep']
    num_actions=len(actions);
    gamma=0.9;
    initial_state=1;

    policy=np.zeros([num_actions,num_states])
    P_a_s_sd=np.zeros([num_actions,num_states,num_states]);

    states=range(num_states);
    rewards=np.zeros([num_states,num_actions])
    #-----------Fill in distributions
    P_a_s_sd[3,4,1]=0.2;
    P_a_s_sd[3,4,2]=0.4;
    P_a_s_sd[3,4,3]=0.4;
    P_a_s_sd[0,0,0]=1;
    P_a_s_sd[1,0,1]=1;
    P_a_s_sd[0,1,0]=1;
    P_a_s_sd[2,1,2]=1;
    P_a_s_sd[2,2,3]=1;
    P_a_s_sd[4,2,5]=1;
    P_a_s_sd[2,3,5]=1;
    P_a_s_sd[3,3,4]=1;

    policy[0,0]=0.5
    policy[1,0]=0.5
    policy[0,1]=0.5
    policy[2,1]=0.5
    policy[2,2]=0.5
    policy[4,2]=0.5
    policy[3,3]=0.5
    policy[4,3]=0.5

    rewards[0,0]=-1;
    rewards[0,1]=0;
    rewards[1,0]=-1;
    rewards[1,2]=-2;
    rewards[2,2]=-2;
    rewards[2,4]=0;
    rewards[3,2]=10;
    rewards[3,3]=1;

    
