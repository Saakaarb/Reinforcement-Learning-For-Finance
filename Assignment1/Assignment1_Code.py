import numpy as np
import math
import itertools
import scipy.stats
import mdp
from matplotlib import pyplot as plt

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
    var=1;
    gamma=0.95;
    #-------
    #Numerical Parameters definition

    W_0=0.2; # Initial Wealth

    W_max=2.0;# Maximum Wealth
    n_w=25;#Divisions of wealth scale

    T=1.0;# End Time
    n_t=25; # Divisions on time scale
    Wi=0.01;#minimum wealth
    

    t=np.linspace(0,T,n_t);
    Wt=np.linspace(Wi,W_max,n_w);
    dWt=10**(-3);
    dt=t[1]-t[0]
    state_space=np.array(list((itertools.product(t,Wt))));
    
    action_space=np.linspace(Wi,W_max,n_w);


#    print(state_space)
    #print(action_space)
    num_actions=action_space.shape[0];
    num_states=state_space.shape[0];
    default_action=1./num_actions
    
    for i_state in range(num_states):
        state=state_space[i_state];
        if near(state[0],0.0):
#            print("Kobe")
            state_space[i_state][1]=W_0;
    
#    print(state_space)
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
                std_current=current_action*(var**0.5);
                #print(mu_current,mu,current_action,current_wealth)
                P_norm=10**(-10);
                for j_state in range(num_states):
                    
                    state_dash=state_space[j_state];
                    next_wealth=state_dash[1];
                    next_time=state_dash[0];
                    
                    if near(next_time,current_time+dt):
                        prob=scipy.stats.norm(loc=mu_current,scale=std_current).cdf(next_wealth+dWt)-scipy.stats.norm(loc=mu_current,scale=std_current).cdf(next_wealth-dWt);
                        P_dict[i_state][i_action][j_state]=prob;
                        P_norm=P_norm+prob;
                #P_dict[i_state][i_action][j_state]=P_dict[i_state][i_action][j_state]/P_norm;
                        
                
                for j_state in range(num_states):
                    state_dash=state_space[j_state];
                    next_wealth=state_dash[1];
                    next_time=state_dash[0];

                    if near(next_time,current_time+dt):
                        P_dict[i_state][i_action][j_state]=P_dict[i_state][i_action][j_state]/P_norm;
                
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
#    print(P);
    for a in range(P.shape[0]):
        for s in range(P.shape[1]):
            print(np.any(P[a,s,:])>0)
    a=1;
#   print(state_space);
#   print(action_space)
    #print(pi_dict);
    #print(pi);
    best_action,vk=mdp.policy_iteration(S,A,P,R,gamma,pi)
    print(best_action) 
    
    avg_best_pol=np.zeros(t.shape[0])-1;

    for i_t in range(t.shape[0]):
        
        sum_t=0;
        t_val=t[i_t]
        counter_t=0;
        for i_state in range(num_states):


            if state_space[i_state][0]==t_val:
        
                sum_t=sum_t+action_space[int(best_action[i_state])];
                counter_t=counter_t+1;
        avg_best_pol[i_t]=sum_t/counter_t;
    print(var,a)    
    true_soln=((mu-r))/(np.power((1+r),(T-1-t))*var*a)

    plt.plot(t[:-1],avg_best_pol[:-1],label="Policy Iteration Solution")
    plt.plot(t[:-1],true_soln[:-1],label="True Solution")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Money given to risky asset")
    plt.title("Optimal Policy Plot");
    plt.show()
            
        


    


