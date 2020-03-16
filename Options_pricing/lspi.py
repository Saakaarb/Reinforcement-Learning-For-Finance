import numpy as np
import math
from matplotlib import pyplot as plt
import random
from copy import deepcopy
from typing import List
def decision(probability:float)-> float:
    return random.random() < probability




def monte_carlo(T:float,dt:float,q:float,S0:float,increase:float,decrease:float)->List[float]:
    # Runs a single monte-carlo simulation

    #Inputs: T: Expiration time
    #dt: Time step
    #q: Risk Neutral Probability 
    #S0: Initial Asset price

    #outputs: Simulated asset price from one rollout
    num_timesteps=int(T/dt);

    SP=np.zeros([num_timesteps+1]);
    SP[0]=S0;
    for i_step in range(num_timesteps):
        
        if decision(q):
            SP[i_step+1]=SP[i_step]*increase;
        else:
            SP[i_step+1]=SP[i_step]*decrease;
    
    return SP

def create_simulatedpaths(T,dt,q,S0,increase,decrease,m)-> List[List[float]]:
    
    #Return square matrix of shape: (num simulated paths, num timesteps);
    
    #Inputs: T: Expiration time
    #dt: Time step
    #q: Risk Neutral Probability
    #S0: Initial Asset price


    num_timesteps=int(T/dt);
    SP_main=np.zeros([m,num_timesteps+1]);
    
    for i_m in range(m):
        SP_main[i_m,:]=deepcopy(monte_carlo(T,dt,q,S0,increase,decrease));
    
    return SP_main;
    

def payoff(K,St):
    # Payoff for a put option
    return max(K-St,0);

def feature_0(K,St):
    return 1
def feature_1(K,St):
    return np.exp(-St/(2*K));
def feature_2(K,St):
    return np.exp(-St/(2*K))*(1-St/K);
def feature_3(K,St):
    return np.exp(-St/(2*K))*(1-2*St/K+((St/K)**2)/2)

def LSPI(T:float,dt:float,q:float,S0:float,increase:float,decrease:float,m:int,batch_size:int,r:float,K:float,num_features:int)->int:

    #Implementation of Least-Squares Policy Iteration to optimally exercise American Put Options

    #Inputs: T: Expiration time
    #dt: Time step
    #q: Risk Neutral Probability
    #S0: Initial Asset price
    #m: Number of Simulation paths to average over
    #r: Rate of riskless return;
    #K: Strike Price

    # Returns whether we should exercise option at t=0
    SP=deepcopy(create_simulatedpaths(T,dt,q,S0,increase,decrease,m))
    discount_factor=np.exp(-r*dt);
    num_timesteps=int(T/dt);
    A=np.zeros([num_features,num_features])
    b=np.zeros([num_features])

    w=np.zeros([num_features]);

    for i_m in range(m): # Loop through simulation paths 
        
        
        for i_time in range(num_timesteps-1):# Loop through time
            
            Q=payoff(K,SP[i_m,i_time+1])
            St=SP[i_m,i_time]
            Stp1=SP[i_m,i_time+1]
            phi_all_next=np.array([feature_0(K,Stp1),feature_1(K,Stp1),feature_2(K,Stp1),feature_3(K,Stp1)])
            phi_all_cur=np.array([feature_0(K,St),feature_1(K,St),feature_2(K,St),feature_3(K,St)])

            if i_time<num_timesteps-1 and Q <= np.dot(w,phi_all_next): # The off-policy is greedy, this condition implies continuation > payoff

                P=deepcopy(phi_all_next)
            else:
                P=np.zeros([phi_all_next.shape[0]])
            if Q > np.dot(w,P):  # If payoff > expected value of continuation
                R=Q
            else:
                R=0
            A=deepcopy(A+np.outer(phi_all_cur,phi_all_cur-discount_factor*P)) # Add the experience to A
            b=deepcopy(b+discount_factor*R*phi_all_cur) # Add the rewards to B
        
    w=deepcopy(np.matmul(np.linalg.inv(A),b)) # Least squares solution to optimal w, which can give us Q(expected returns from a state)
    St=np.mean(SP[:,0]) # Underlying stock price at initial state
    phi_0=[feature_0(K,St),feature_1(K,St),feature_2(K,St),feature_3(K,St)] # feature functions

    exercise=payoff(K,S0)  # Payoff of immediate exercise
    continue_val=discount_factor*np.dot(w,phi_0) # Expectation of continuing
    print(exercise,continue_val) # compare for t=0, can be done for any time based on simulation paths
    if exercise>continue_val:
        return 1;
    else:
        return 0;

    
if __name__=="__main__":

    T=1.5;
    dt=0.5;
    increase=1.2;
    decrease=0.8;
    m=60;
    batch_size=5;
    r=0.05;
    K=100;
    S0=100;
    num_features=4;
    discount_factor=np.exp(-r*dt);
    q=(discount_factor**(-1)-decrease)/(increase-decrease)# Risk Neutral Probability measure, that stock increases in value
    
    current_action=LSPI(T,dt,q,S0,increase,decrease,m,batch_size,r,K,num_features)
