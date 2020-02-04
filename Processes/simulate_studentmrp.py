import numpy as np
import math
import mrp

if __name__=="__main__":


    states=np.array(range(7))
    num_states=states.shape[0];
    gamma=0.9;

    P_s_sd=np.zeros([num_states,num_states]);
    r_s_sd=np.zeros([num_states,num_states]);# not using create_mrp function here

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

    V=mrp.solve_MRP(states,P_s_sd,rewards,gamma) # Solve using matrix definition
    print(V);


#--------------


