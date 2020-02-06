import numpy as np
from binarytree import Node
import binarytree
def create_subtree(root_tree,val_left,val_right,increase,decrease,depth):
    #Recurse to create binary tree
    #Output: binary tree of states
    if depth==0:
        return root_tree
    else:
        
        root_tree.right=Node(val_right);
        root_tree.left=Node(val_left);
        root_tree_1=create_subtree(root_tree.right,val_right*decrease,val_right*increase,increase,decrease,depth-1);
        root_tree_2=create_subtree(root_tree.left,val_left*decrease,val_left*increase,increase,decrease,depth-1);
    
    return root_tree_2

    
def payoff(K,St):
    # Payoff of a put option

    #K:
    return max(K-St.value,0);

def optimal_policy(root,r,dt,q,depth):

    # Find optimal action at each time state given a payoffs tree, risk-neutral probabilities and discount factor
    
    #Inputs:
    #root: Payoff value tree
    #r: risk free rate
    #dt: time interval b/w option exercise
    #q: risk neutral probability of asset increasing in price
    #depth: depth of tree
    
    #returns:
    #action_tree: binary tree of optimal actions for non-leaf state nodes
    #expectation_tree:  binary tree of expected optimal value functions at (t+1) for non-leaf state nodes
    
    num_parents=len(root)-root.leaf_count;
    
    list_vals=root.values;
    expected_payoffs=[];
    optimal_action=[];
    discount_factor=np.exp(-r*dt)
    for i in range(num_parents):
        exp_payoff=(q*list_vals[2*i+2]+(1-q)*list_vals[2*i+1])*discount_factor 
        expected_payoffs.append(exp_payoff);
        action=0;
        if list_vals[i]>exp_payoff:
            action=1;
        optimal_action.append(action);

    action_tree=binarytree.build(optimal_action);
    expectation_tree=binarytree.build(expected_payoffs)
    return action_tree,expectation_tree


def americanopt_put_tree(S0,time,dt,increase,decrease,strike_price,riskfree_rate):
    #Creates a binary tree for an american put option, finds optimal action(exercise option or not) for a put option

    #S0: Price of underlying stock at t=0;
    #time: total time
    #dt: time intervals at which option can be exercised
    #increase: factor of increase of St at every time step
    #decrease: factor of decrease of St at every time step
    #strike_price:strike price of put option
    
    discount_factor=np.exp(-r*dt);
    q=(discount_factor**(-1)-decrease)/(increase-decrease)# Risk Neutral Probability measure, that stock increases in value

    depth=int(time/dt);
    #depth=2;    

    root=Node(S0)
    
    create_subtree(root,S0*decrease,S0*increase,increase,decrease,depth)
    print("Binary tree:")
    print(root) # Print Binary Tree
    
    for i_root in range(len(root.levelorder)):

        root.levelorder[i_root].value=payoff(strike_price,root.levelorder[i_root]);
    print("Payoffs:")
    print(root) # Print Payoffs
   
    
    action_tree,expectation_tree=optimal_policy(root,r,dt,q,depth) 
    print("Expectations from Bellman Equation:")
    print(expectation_tree)
    
    print("Therefore, optimal actions:")
    print(action_tree)
    

if __name__=="__main__":

    S0=100;
    time=1.5;
    dt=0.5;
    increase=1.2;
    decrease=0.8;
    strike_price=100;
    r=0.05;
    americanopt_put_tree(S0,time,dt,increase,decrease,strike_price,r);

