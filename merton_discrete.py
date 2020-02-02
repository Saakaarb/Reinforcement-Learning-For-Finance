import numpy as np
import math
import itertools
import scipy.stats
import mdp
def near(a1,a2):

    tol=10**(-4);
    if np.abs(a1-a2)<tol:
        return True
    else:
        return False
    


if __name__=="__main__":

    #Problem Parameters Definition:
    a=1;
    mu=0.2;
    r=0.05;
    var=0.2;
    gamma=0.95;
    #-------
    #Numerical Parameters definition

    W_0=2.0;

    W_max=10.0;
    n_w=11;

    T=1.0;
    n_t=11;
    Wi=0.1;
    default_action=W_0/10;

    t=np.linspace(0,T,n_t);
    Wt=np.linspace(Wi,W_max,n_w);
    dWt=Wt[1]-Wt[0];
    state_space=np.array(list((itertools.product(t,Wt))));
    action_space=np.linspace(Wi,W_max,n_w);

    num_actions=action_space.shape[0];
    num_states=state_space.shape[0];

    #Reward Dictionary
    R_dict={}
    for i_state in range(num_states):
        
        state=state_space[i_state];
        R_dict[i_state]={}
        if near(state[0],T):
                
            for i_action in range(num_actions):
                R_dict[i_state][i_action]=-np.exp(-a*state[1])/a;

        else:

            for i_action in range(num_actions):
                R_dict[i_state][i_action]=0;

    #--------------------

    #Transition Probability

    P_dict={}

    for i_state in range(num_states):

        state=state_space[i_state];
        
        current_wealth=state[1];
        current_time=state[0];
        P_dict[i_state]={};

        for i_action in range(num_actions):
            current_action=action_space[i_action];

            P_dict[i_state][i_action]={};

            if current_action>current_wealth:
                continue;
            else:
                
                mu_current=current_action+current_action*mu+(1+r)*(current_wealth-current_action);
                std_current=current_action*var**0.5;

                for j_state in range(num_states):
                    
                    state_dash=state_space[j_state];
                    next_wealth=state_dash[1];
                    next_time=state_dash[0];
                    
                    if near(next_time,current_time+1):
                        prob=scipy.stats.norm(loc=mu_current,scale=std_current).cdf(next_wealth+dWt)-scipy.stats.norm(loc=mu_current,scale=std_current).cdf(next_wealth);
                        P_dict[i_state][i_action][j_state]=prob;

    #--------------
    #pi dictionary:Initialize Random Policy

    pi_dict={}
    
    for i_state in range(num_states):

        pi_dict[i_state]={};
        state=state_space[i_state];
        current_wealth=state[1];
        current_time=state[0];

        if near(current_time,T):
            continue;
        else:    
            for i_action in range(num_actions):
                pi_dict[i_state][i_action]=default_action;
            


    [S,A,P,R,gamma,pi]=mdp.create_MDP(state_space,action_space,P_dict,R_dict,gamma,pi_dict);
    best_action,vk=mdp.value_iteration(S,A,P,R,gamma,pi)
    print(best_action)
