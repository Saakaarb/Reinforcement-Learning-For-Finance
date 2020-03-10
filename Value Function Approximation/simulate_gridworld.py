import numpy as np
import mdp
from collections import defaultdict
from copy import deepcopy
from mdp_sampler import mdp_sampler
from backward_td_lambda_vfa import  bwd_td_lambda_vfa 
from matplotlib import pyplot as plt
from sarsa_vfa import sarsa_lambda_vfa
from feature_functions import feature_functions
from monte_carlo_vfa import monte_carlo
from sa_features import feature_functions_sa
def rms_err(a1,a2):

    return np.power(np.sum(np.power((a1-a2),2)),0.5)
def get_actions(Q,P):

    state_best_action=[]
    state_best_valfn=[]
    for i_state in range(Q.shape[0]):
        valfn_comp=float("-inf")
        best_action=-1;
        if (P[:,i_state,:]==0).all():
                state_best_action.append(Q.shape[1])
                state_best_valfn.append(0)
                continue

        for i_action in range(Q.shape[1]):
            
            if np.any(P[i_action,i_state,:]>0):

                if Q[i_state,i_action]> valfn_comp:
                    valfn_comp=Q[i_state,i_action]
                    best_action=i_action
        state_best_action.append(best_action)
        state_best_valfn.append(valfn_comp)

    return state_best_action,state_best_valfn

if __name__=="__main__":

    
    gamma=0.9;

    S=np.array(range(16));# States
    A=np.array(range(4));#Actions: referenced by integers,  0: up, 1: right, 2: down, 3:left, 4: stay
    terminal_states=[0,15]    


    R_dict={
            0:{},
            1:{1:-1,2:-1,3:-1},
            2:{1:-1,2:-1,3:-1},
            3:{2:-1,3:-1},
            4:{0:-1,1:-1,2:-1},
            8:{0:-1,1:-1,2:-1},
            12:{0:-1,1:-1},
            7:{0:-1,2:-1,3:-1},
            11:{0:-1,2:-1,3:-1},
            15:{},
            13:{0:-1,1:-1,3:-1},
            14:{0:-1,1:-1,3:-1},
            5:{0:-1,1:-1,2:-1,3:-1},
            6:{0:-1,1:-1,2:-1,3:-1},
            9:{0:-1,1:-1,2:-1,3:-1},
            10:{0:-1,1:-1,2:-1,3:-1},
            
            
            }

    pi_dict={
            0:{0:0,1:0,2:0,3:0 },
            1:{0:0,1:0.333333333333, 2:0.333333333333, 3:0.333333333333 },
            2:{0:0 ,1:0.333333333333, 2:0.333333333333, 3:0.333333333333 },
            3:{0:0 ,1:0, 2:0.5, 3:0.5 },
            4:{0:0.333333333333, 1:0.333333333333, 2:0.333333333333, 3:0 },
            5:{0:0.25, 1:0.25, 2:0.25, 3:0.25 },
            6:{0:0.25, 1:0.25, 2:0.25, 3:0.25 },
            9:{0:0.25, 1:0.25, 2:0.25, 3:0.25 },
            10:{0:0.25, 1:0.25, 2:0.25, 3:0.25 },
            7:{0:0.333333333333 ,1:0., 2:0.333333333333, 3:0.333333333333 },
            8:{0:0.333333333333, 1:0.333333333333, 2:0.333333333333, 3:0 },
            11:{0:0.333333333333, 1:0., 2:0.333333333333, 3:0.333333333333 },
            12:{0:0.5, 1:0.5, 2:0., 3:0. },
            15:{0:0, 1:0., 2:0., 3:0 },
            13:{0:0.333333333333, 1:0.333333333333, 2:0., 3:0.333333333333 },
            14:{0:0.333333333333, 1:0.333333333333, 2:0., 3:0.333333333333 },

            
            }
    #print(pi_dict)
    #Create P_dict
    P_dict={}
    for key1,val1 in pi_dict.items():
        P_dict[key1]={}

        for key2 in val1:
            P_dict[key1][key2]={}


    for key1,val1 in pi_dict.items():

        for key2 in val1:

            if pi_dict[key1][key2]>0:

                if key2==0:
                    key3=key1-4;
                elif key2==1:
                    key3=key1+1;
                elif key2==2:
                    key3=key1+4;
                elif key2==3:
                    key3=key1-1;
                else:
                    key3=key1;

                P_dict[key1][key2][key3]=1;
    
    [S,A,P,R,gamma,pi]=mdp.create_MDP(S,A,P_dict,R_dict,gamma,pi_dict) # Creates the relevant matrices for the gridworld
    #print(P,R,pi)
    #print(P[0,0,:])
    sampler=mdp_sampler(S,A,P,pi,R)
    dp_solution=mdp.evaluate_policy(S,A,P,R,gamma,pi)
    
    #--------------------------------------------------Model Free Prediction(VFA)
    
    # Monte Carlo Method
    
    #lambd=[0.1*(i) for i in range(num_lambda)]

    alpha=10**(-1)
    n_episodes=100000
    num_features=3
    [w_on]=monte_carlo(gamma,alpha,n_episodes,S,num_features,A,sampler)
    print(w_on)
    features=feature_functions("lagrange",num_features,S)
    value_fn=[]
    for i_state in range(len(S)):

        current_features=features.get_featvector(i_state)

        value_fn.append(np.dot(w_on,current_features))

    print(value_fn)
    plt.plot(S,dp_solution)
    plt.plot(S,value_fn)
    plt.xlabel("State")
    plt.ylabel("Value function")
    plt.title("Comparison of DP, Monte-Carlo")
    plt.show()
    
    #--------------------------------------------------
    # TD-lambda method
    ''' 
    #define grid of parameters for learning
    num_lambda=1
    #lambd=[0.1*(i) for i in range(num_lambda)]
    lambd=0.9
    
    alpha=10**(-3)
    n_episodes=1200
    num_features=3
    rms_error=[]
    [w_on,E]=bwd_td_lambda_vfa(gamma,lambd,alpha,n_episodes,S,num_features,A,sampler,mode="online")
    print(w_on) 
    features=feature_functions("lagrange",num_features,S)
    value_fn=[]
    for i_state in range(len(S)):
        
        current_features=features.get_featvector(i_state)

        value_fn.append(np.dot(w_on,current_features))

    print(value_fn)
    plt.plot(S,dp_solution)
    plt.plot(S,value_fn)
    #plt.legend(('Dynamic Programming','Online TD Lambda','Offline TD Lambda'))
    #plt.show()
    #print(V,dp_solution)
    plt.xlabel("State")
    plt.ylabel("Value function")
    plt.title("Comparison of DP, online TD Lambda")
    plt.show()    
    '''
      
    #--------------------------------------------------Model Free control(VFA)
    '''
    num_features=3
    features= feature_functions_sa("lagrange",num_features,S,A)
    funcs=features.feature_vals

     
    #--------------------------------
    # SARSA-LAMBDA with VFA
    lambd=0.9
    alpha=10**(-4)
    n_episodes=2000
    eps=0.1
    [w,E]=sarsa_lambda_vfa(gamma,lambd,alpha,eps,n_episodes,S,num_features,A,sampler)    
    print(w)
    learned_V=[]   
    best_pol=[]
    for i_state in range(len(S)):
        best_Q=np.float("-inf")
        print("state= %d"% i_state)
        for i_action in range(len(A)):
            current_features=features.get_featvector(i_state,i_action)
            Q_sa=np.dot(current_features,w)
            
            print(Q_sa)
            if Q_sa > best_Q:  # equivalent to taking an argmax
                best_a=i_action
                best_Q=Q_sa
        learned_V.append(best_Q)
        best_pol.append(best_a)
    
    best_action_2,dp_solution_q=mdp.value_iteration(S,A,P,R,gamma,pi);
    print(best_pol)
    print(best_action_2)
    plt.plot(S,learned_V)
    plt.plot(S,dp_solution_q)
    plt.show()
    
    #best_action_extract
    #best_actions,best_valfn=get_actions(Q,P)
    #print(best_valfn)
    '''
    '''
    fig2,ax2=plt.subplots()
    ax2.plot(S,best_valfn,marker='x')
    ax2.plot(S,dp_solution_q)
    ax2.legend(('Q-LAMBDA','Value Iteration'))
    plt.xlabel('State Index')
    plt.ylabel('Optimal Value Function V*')
    plt.title('Comparison of Q-LAMBDA and value iteration for gridworld ')
    plt.show()
    '''
