import numpy as np
import math
from matplotlib import pyplot as plt
import random
from copy import deepcopy

def decision(probability):
    return random.random() < probability


def monte_carlo(T,dt,q,S0,increase,decrease):
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

def create_simulatedpaths(T,dt,q,S0,increase,decrease,m):
    
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
    

#def payoff(K,St):
    # Payoff for a call option
#    return max(St-K,0);


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

def longstaff_schwartz(T,dt,q,S0,increase,decrease,m,r,K,num_features):

    #Implementation of Longstaff-Schwartz Algorithm to optimally exercise American Options

    #Inputs: T: Expiration time
    #dt: Time step
    #q: Risk Neutral Probability
    #S0: Initial Asset price
    #m: Number of Simulation paths to average over
    #r: Rate of riskless return;
    #K: Strike Price
    SP=deepcopy(create_simulatedpaths(T,dt,q,S0,increase,decrease,m))
    #print(SP) 
    discount_factor=np.exp(-r*dt);
    num_timesteps=int(T/dt);
    CF=np.zeros([m]);
    #X=np.zeros([m,num_features])
    #Y=np.zeros([m])
    weights=np.zeros([num_features]);
    for i in range(CF.shape[0]):

        CF[i]=payoff(K,SP[i,-1]);# Expiration time option mandatory excercise payoff.
    
    for i_time in range(num_timesteps-1,0,-1): # Loop between times last, and first non inclusive
        X=[]
        Y=[]
        CF=CF*discount_factor;
        t=i_time*dt
        for i_m in range(m):
            
            if payoff(K,SP[i_m,i_time])>0:
                #X[i_m,0]=feature_0(T,t);
                #X[i_m,1]=feature_1(T,t);
                #X[i_m,2]=feature_2(T,t);
                St=SP[i_m,i_time];
                Y.append(CF[i]);
                X.append([feature_0(K,St),feature_1(K,St),feature_2(K,St),feature_3(K,St)]);
        X=np.array(X);
        Y=np.array(Y);
        #print(X)
        weights=deepcopy(np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.matmul(np.transpose(X),Y))); # Regression solution
        
        for i in range(Y.shape[0]):

            if payoff(K,SP[i,i_time])>np.sum(np.multiply(weights,X[i,:])): # If condition is true, option should be exercised
                CF[i]=payoff(K,SP[i,i_time]);   # Cashflow in exercising option
    exercise=payoff(K,S0);  # Payoff of immediate exercise
    continue_val=discount_factor*np.mean(CF) # Expectation of cashflow
    print(exercise,continue_val)
    if exercise>continue_val:
        return 1;
    else:
        return 0;

    
if __name__=="__main__":

    T=1.5;
    dt=0.5;
    increase=1.2;
    decrease=0.8;
    m=20;
    r=0.05;
    K=100;
    S0=100;
    num_features=3;
    discount_factor=np.exp(-r*dt);
    q=(discount_factor**(-1)-decrease)/(increase-decrease)# Risk Neutral Probability measure, that stock increases in value
    
    current_action=longstaff_schwartz(T,dt,q,S0,increase,decrease,m,r,K,num_features)
