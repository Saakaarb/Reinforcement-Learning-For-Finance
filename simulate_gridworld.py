import numpy as np
import mdp
from collections import defaultdict
from copy import deepcopy
if __name__=="__main__":

    
    gamma=1;

    S=np.array(range(16));# States
    A=np.array(range(5));#Actions: referenced by integers,  0: up, 1: right, 2: down, 3:left, 4: stay
    


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
            0:{0:0,1:0,2:0,3:0,4:1 },
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
            15:{0:0, 1:0., 2:0., 3:0, 4:1 },
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
    
    vi=mdp.evaluate_policy(S,A,P,R,gamma,pi);# Evaluates Initial random policy, Pg 12 of DP lecture
    print(vi)                            # Printed values are different due to round offs in lecture numbers
    best_action,vk=mdp.policy_iteration(S,A,P,R,gamma,pi);# Policy Iteration to find the best policy
    print(best_action)                   # Acc to the code, it finds only 1 best action for a given state, in decreasing priority of 
<<<<<<< HEAD
                                                                                                                        #action vector
    best_action_2,vk=mdp.value_iteration(S,A,P,R,gamma,pi);
    print(best_action_2)
=======
                                                                                                                    #action vector

>>>>>>> 513a8ceb6a5718104300190a413e7ddbfb748a2a
