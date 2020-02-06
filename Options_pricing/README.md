The folder contains several files related to options pricing assignments.

**btree_american_opt.py**:  
This contains code to solve an american option problem: when to optimally exercise option/ arbitrage free pricing method.  
To run the code, the library "binarytree" must be downloaded. Code works for a general problem, if the stock price can be modelled by an increasing/decreasing factor every time.   

The code with current parameters solves for a put option problem, with the following parameters:  
    Initial asset price=100;  
    time to expiry=1.5 units;  
    time intervals at which option can be exercised=0.5;  
    ratio of potential increase of asset price=1.2;  
    ratio of potential decrese of asset price=0.8;  
    strike_price=100;  
    risk free return rate=0.05;  
    
 The problem solution is as shown below:
 
 ![Screenshot1](results_screenshot.png)
 
 The problem parameters and solution is corroborated by the tutorial problem at the following link: https://www.youtube.com/watch?v=35n7TICJbLc


